# python train.py --model EEGNet --epochs 700 --plot --no-init_weights
# python train.py --model SSVEPNet --epochs 700 --no-compile --plot
python train.py --model Deformer --epochs 800 --no-init_weights --plot

# python train.py --model EEGNet --epochs 700 --data_path datasets/Noisy-UTEC --dataset UTEC --params_path datasets/UTEC.yaml --no-init_weights --plot
# python train.py --model SSVEPNet --epochs 700 --data_path datasets/Noisy-UTEC --dataset UTEC --params_path datasets/UTEC.yaml --no-compile --plot
python train.py --model Deformer --epochs 800  --data_path datasets/Noisy-UTEC --dataset UTEC --params_path datasets/UTEC.yaml --no-init_weights --plot