from training.transformer_training_autocorr import transformer_train_autocorr

transformer_train_autocorr(num_epochs=1000,
                           num_patients=None,
                           batch_size=512,
                           hpc=True)

print('Finished executing main.py.')


