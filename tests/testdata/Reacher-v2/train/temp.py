import tensorflow as tf

# Path to the original checkpoint (index and data files)
old_checkpoint_path = 'policy/ckpt-255000'

# Create a checkpoint reader
ckpt_reader = tf.train.load_checkpoint(old_checkpoint_path)

# Get a list of all variable names and shapes from the checkpoint
variable_names = ckpt_reader.get_variable_to_shape_map()

print(variable_names)

# # Create a dictionary to hold the renamed variables
# renamed_variables = {}

# # Define the variable mapping (old name -> new name)
# var_mapping = {
#     'policy/_wrapped_policy/_actor_network/_encoder/_postprocessing_layers/2/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'policy/_wrapped_policy/_actor_network/_encoder/2/kernel/.ATTRIBUTES/VARIABLE_VALUE',
#     'policy/_wrapped_policy/_actor_network/_encoder/_postprocessing_layers/2/bias/.ATTRIBUTES/VARIABLE_VALUE': 'policy/_wrapped_policy/_actor_network/_encoder/2/bias/.ATTRIBUTES/VARIABLE_VALUE',
#     'policy/_wrapped_policy/_actor_network/_encoder/_postprocessing_layers/1/kernel/.ATTRIBUTES/VARIABLE_VALUE': 'policy/_wrapped_policy/_actor_network/_encoder/1/kernel/.ATTRIBUTES/VARIABLE_VALUE',
#     'policy/_wrapped_policy/_actor_network/_encoder/_postprocessing_layers/1/bias/.ATTRIBUTES/VARIABLE_VALUE': 'policy/_wrapped_policy/_actor_network/_encoder/1/bias/.ATTRIBUTES/VARIABLE_VALUE',
# }   


# # Create new variables by copying values from the old checkpoint and renaming
# for old_name, new_name in var_mapping.items():
#     if old_name in variable_names:
#         # Get the tensor values from the checkpoint
#         tensor_value = ckpt_reader.get_tensor(old_name)
        
#         # Create a new variable with the renamed name
#         renamed_variables[new_name] = tf.Variable(tensor_value, name=new_name)

#     else:
#         tensor_value = ckpt_reader.get_tensor(old_name)
#         renamed_variables[old_name] = tf.Variable(tensor_value, name=old_name)

# # Now create a Checkpoint object with the renamed variables
# checkpoint = tf.train.Checkpoint(**renamed_variables)

# # Save the new checkpoint
# new_checkpoint_path = 'new_policy/ckpt-255000'
# checkpoint.save(new_checkpoint_path)

# ckpt_reader = tf.train.load_checkpoint(new_checkpoint_path)

# # Get a list of all variable names and shapes from the checkpoint
# variable_names = ckpt_reader.get_variable_to_shape_map()

# print(variable_names)

