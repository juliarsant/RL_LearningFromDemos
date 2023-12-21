from hitl_train import run
from train import run_train
from data import plot
from imports import trials

human = True
run_all = True

if __name__=="__main__":
    if run_all == True:
        run_train(trials=trials)
        run(trials=trials)
        plot()
    elif human == True:
        run(trials=trials)
        plot()
    else:
        run_train(trials=trials)

