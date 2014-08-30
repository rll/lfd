from __future__ import division

import trajoptpy, bulletsimpy
import numpy as np

import IPython as ipy

class SimulationObject(object):
    def __init__(self, names, dynamic=True):
        self.names = names
        self.dynamic = dynamic
        self.add_after = False # if True, this object needs to be added after the BulletEnvironment has been created
        self.sim_env = None

    def add_to_env(self, sim_env):
        raise NotImplementedError
    
    def remove_from_env(self):
        raise NotImplementedError
    
    def get_bullet_objects(self):
        if self.sim_env is None:
            raise RuntimeError("get_bullet_objects should only be called when the object is in an environment")
        bt_objs = []
        for name in self.names:
            body = self.sim_env.env.GetKinBody(name)
            bt_obj = self.sim_env.bt_env.GetObjectFromKinBody(body)
            bt_objs.append(bt_obj)
        return bt_objs
    
    def get_state(self):
        return np.asarray([bt_obj.GetTransform() for bt_obj in self.get_bullet_objects()])
    
    def set_state(self, tfs):
        for (bt_obj, tf) in zip(self.get_bullet_objects(), tfs):
            bt_obj.SetTransform(tf)

class XmlSimulationObject(SimulationObject):
    def __init__(self, xml, dynamic=False):
        super(XmlSimulationObject, self).__init__(None, dynamic=dynamic) # TODO: None names
        self.xml = xml

    def add_to_env(self, sim_env):
        self.sim_env = sim_env
        pre_names = [body.GetName() for body in self.sim_env.env.GetBodies()]
        if '<' in self.xml: # TODO: fix this hack
            self.sim_env.env.LoadData(self.xml)
        else:
            self.sim_env.env.Load(self.xml)
        post_names = [body.GetName() for body in self.sim_env.env.GetBodies()]
        self.names = [name for name in post_names if name not in pre_names]
    
    def remove_from_env(self):
        for bt_obj in self.get_bullet_objects():
            self.sim_env.env.Remove(bt_obj.GetKinBody())
        self.sim_env = None
    
    def __repr__(self):
        return "XmlSimulationObject(%s, dynamic=%r)" % (self.xml, self.dynamic)

class BoxSimulationObject(XmlSimulationObject):
    def __init__(self, name, translation, extents, dynamic=True):
        xml = """
        <Environment>
          <KinBody name="%s">
            <Body type="%s" name="%s_link">
              <Translation>%f %f %f</Translation>
              <Geom type="box">
                <extents>%f %f %f</extents>
              </Geom>
            </Body>
          </KinBody>
        </Environment>
        """ % (name, 'dynamic' if dynamic else 'static', name, translation[0], translation[1], translation[2], extents[0], extents[1], extents[2])
        super(BoxSimulationObject, self).__init__(xml, dynamic=dynamic)
        self.name = name
        self.translation = translation
        self.extents = extents
    
    def __repr__(self):
        return "BoxSimulationObject(%s, %s, %s, dynamic=%r)" % (self.name, self.translation, self.extents, self.dynamic)

class CylinderSimulationObject(XmlSimulationObject):
    def __init__(self, name, translation, radius, height, dynamic=True):
        xml = """
        <Environment>
          <KinBody name="%s">
            <Body type="%s" name="%s_link">
              <Translation>%f %f %f</Translation>
              <Geom type="cylinder">
                <rotationaxis>1 0 0 90</rotationaxis>
                <radius>%f</radius>
                <height>%f</height>
              </Geom>
            </Body>
          </KinBody>
        </Environment>
        """ % (name, 'dynamic' if dynamic else 'static', name, translation[0], translation[1], translation[2], radius, height)
        super(CylinderSimulationObject, self).__init__(xml, dynamic=dynamic)
        self.name = name
        self.translation = translation
        self.radius = radius
        self.height = height
    
    def __repr__(self):
        return "CylinderSimulationObject(%s, %s, %s, %s, dynamic=%r)" % (self.name, self.translation, self.radius, self.height, self.dynamic)

class RopeSimulationObject(SimulationObject):
    def __init__(self, name, ctrl_points, rope_params, dynamic=True, upsample=0, upsample_rad=1):
        super(RopeSimulationObject, self).__init__([name], dynamic=True)
        self.add_after = True
        self.name = name
        self.init_ctrl_points = ctrl_points
        self.rope_params = rope_params
        self.capsule_rope_params = bulletsimpy.CapsuleRopeParams()
        self.capsule_rope_params.radius       = self.rope_params.radius
        self.capsule_rope_params.angStiffness = self.rope_params.angStiffness
        self.capsule_rope_params.angDamping   = self.rope_params.angDamping
        self.capsule_rope_params.linDamping   = self.rope_params.linDamping
        self.capsule_rope_params.angLimit     = self.rope_params.angLimit
        self.capsule_rope_params.linStopErp   = self.rope_params.linStopErp
        self.capsule_rope_params.mass         = self.rope_params.mass
        self.upsample = upsample
        self.upsample_rad = upsample_rad
        self.rope = None

    def add_to_env(self, sim_env):
        self.sim_env = sim_env
        self.rope = bulletsimpy.CapsuleRope(self.sim_env.bt_env, self.name, self.init_ctrl_points, self.capsule_rope_params)
    
    def remove_from_env(self):
        # remove all capsule-capsule exclude to prevent memory leak
        # TODO: only interate through the capsule pairs that actually are excluded
        cc = trajoptpy.GetCollisionChecker(self.sim_env.env)
        for rope_link0 in self.rope.GetKinBody().GetLinks():
            for rope_link1 in self.rope.GetKinBody().GetLinks():
                cc.IncludeCollisionPair(rope_link0, rope_link1)
        self.sim_env.env.Remove(self.rope.GetKinBody())
        self.rope = None
        self.sim_env = None
    
    def get_bullet_objects(self):
        # method of parent class doesn't work because self.rope casted to BulletObject
        if self.sim_env is None:
            raise RuntimeError("get_bullet_objects should only be called when the object is in an environment")
        return [self.rope]
    
    def get_state(self):
        trans, rots = self.rope.GetTranslations(), self.rope.GetRotations()
        tfs = np.zeros((len(trans), 4, 4))
        tfs[:,:3,3] = trans
        tfs[:,:3,:3] = rots
        tfs[:,3,3] = np.ones(len(trans))
        return tfs
    
    def set_state(self, tfs):
        self.rope.SetTranslations(tfs[:,:3,3])
        self.rope.SetRotations(tfs[:,:3,:3])
    
    def __repr__(self):
        return "RopeSimulationObject(%s, numpy.array([[...]]), RopeParams(...), dynamic=%r, upsample=%i, upsample_rad=%i)" % (self.name, self.dynamic, self.upsample, self.upsample_rad)
