def show_with_prefix(prefix):
    for k in f.keys():
        if k.startswith(prefix):
            print k
            plt.imshow(f[k]['rgb'], interpolation='nearest')
            plt.show()

def rename_group4(group_idx, prefix):
    count = 0
    for i in group_idx:
        f.move(orig_keys[i], prefix + str(count).zfill(2))
        count += 1
