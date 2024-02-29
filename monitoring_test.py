from monitoring import monitor

if __name__=="__main__":
    weights='runs/train/exp/weights/best.pt'
    source='https://www.youtube.com/watch?v=ByED80IKdIU'
    
    for traffic_flow in monitor(weights=weights, source=source):
        print(traffic_flow)