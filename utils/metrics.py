def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.4, 0.6]  # weights for [m_iou, pix_acc] 
    return (x * w).sum(0)