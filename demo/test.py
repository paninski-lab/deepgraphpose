from deepgraphpose.models.fitdgp import fit_dgp
import tensorflow as tf


def run_test(dlcpath, shuffle, batch_size, snapshot):
    step = 2
    gm2, gm3 = 1, 3
    fit_dgp(snapshot,
            dlcpath,
            batch_size=batch_size,
            shuffle=shuffle,
            step=step,
            maxiters=5,
            displayiters=1,
            gm2=gm2,
            gm3=gm3)


if __name__ == '__main__':
    # print(tf.__version__)
    dlcpath = "/Volumes/paninski-locker/data/track_graph3d/bird1-selmaan-2030-01-01"
    shuffle = 1
    batch_size = 1
    snapshot = 'snapshot-step0-final--0'
    run_test(dlcpath, shuffle, batch_size, snapshot)