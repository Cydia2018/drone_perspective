def speedControl(boxCenter):
    x = boxCenter[0]
    y = boxCenter[1]
    actions = []
    if x==480 and y==360:
        actions.append(0)
        actions.append(0)
    elif x==480 and y<360:
        actions.append(0)
        actions.append(20)
    elif x==480 and y>360:
        actions.append(0)
        actions.append(-20)
    elif y==360 and x>480:
        actions.append(20)
        actions.append(0)
    elif y==360 and x<480:
        actions.append(-20)
        actions.append(0)
    elif x>480 and y<360:
        actions.append(20)
        actions.append(20)
    elif x>480 and y>360:
        actions.append(20)
        actions.append(-20)
    elif x<480 and y<360:
        actions.append(-20)
        actions.append(20)
    else:
        actions.append(-20)
        actions.append(-20)

    return actions


def speedControlLinear(boxCenter):
    BASE = 60
    x = boxCenter[0]
    y = boxCenter[1]
    actions = []
    actions.append(round((x - 480) / 480 * BASE))
    actions.append(round((360 - y) / 360 * BASE))

    # if x==480 and y==360:
    #     actions.append(0)
    #     actions.append(0)
    # elif x==480:
    #     actions.append(0)
    #     actions.append(round((360-y)/360*BASE))
    # elif y==360:
    #     actions.append(round((x-480)/480*BASE))
    #     actions.append(0)
    # elif x>480:
    #     actions.append(round((x-480)/480*BASE))
    #     actions.append(round((360-y)/360*BASE))
    # elif x<480:
    #     actions.append(round((x-480)/480*BASE))
    #     actions.append(round((360-y)/360*BASE))
    # else:
    #     pass

    return actions
