from training.transformer_training_autocorr import transformer_train_autocorr

transformer_train_autocorr(num_epochs=10,
                           num_patients=10,
                           batch_size=512,
                           hpc=True,
                           file_format="HDF5")

print('Finished executing main_transformer_raw.py.')


