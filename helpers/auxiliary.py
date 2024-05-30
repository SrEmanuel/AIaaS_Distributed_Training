def contains(list, filter, booleanValue=True):
    for x in list:
        if filter(x):
            return True if booleanValue else x
    return False