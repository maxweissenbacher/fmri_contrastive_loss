from training.nn_training_autocorr import nn_train_autocorr

nn_train_autocorr(
    num_epochs=500,
    num_patients=100,
    batch_size=512,
    hpc=True,
    file_format='HDF5',
)

print('Finished executing.')


