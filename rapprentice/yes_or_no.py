def yes_or_no(question):
    assert isinstance(question,str) or isinstance(question,unicode)
    while True:
        yn = raw_input(question + " (y/n): ")
        if yn=='y': return True
        elif yn=='n': return False