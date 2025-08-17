# EfficientNet-b2 model

## Model inference

To inference model make sure that values of your image is in range [0..1] and 2x256x256 (CxHxW). Model was trained using float16.

## Weights

.pt has folowing structure:
	- epoch - epoch with best validation loss value\
	- state_dict - models weights.\
	- optimizer_state_dict - optimizers parameters\
