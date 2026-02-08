python eval.py --config configs/default.py \
  --checkpoint checkpoints/epoch_20.pkl \
  --clean_root dataset/val/clean --noisy_root dataset/val/noisy \
  --save_dir outputs/val_vis --num_save 5