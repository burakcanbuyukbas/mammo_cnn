def scheduler(epoch, initlr):
    if epoch < 5:
        return initlr
    elif epoch >= 5 > 10:
        return initlr * 0.1
    elif epoch >= 10 > 15:
        return initlr * 0.01
    elif epoch >= 15 > 20:
        return initlr * 0.001
    else:
        return initlr * 0.0001

