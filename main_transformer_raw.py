from training.transformer_training import transformer_train

transformer_train(num_epochs=50,
                  num_patients=10,
                  batch_size=256,
                  hpc=True)

print('Finished executing main_transformer_raw.py.')


