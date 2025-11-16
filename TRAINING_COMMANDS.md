# 🚀 Manual Training Commands - Quick Reference

## Your Dataset
- **Real Images**: 156 photos from your PIC folder
- **AI Images**: 3000 from archive folder
- **Balanced Dataset**: Will use 156 of each class

---

## ⚡ STEP 1: Prepare Dataset (Run Once)

### Option A: Use Batch File (Easiest)
```bash
PREPARE_AND_TRAIN.bat
```

### Option B: Manual Python
```bash
# Install extra packages
pip install pillow-heif psutil

# Prepare dataset
python prepare_for_training.py
```

This will:
- ✅ Copy real images from PIC/ to dataset/real/
- ✅ Convert HEIC files to JPG
- ✅ Copy AI images from archive/ to dataset/fake/
- ✅ Show you the training commands

---

## 🎯 STEP 2: Train the Model (Choose One)

### Quick Test (20-25 minutes)
```bash
python train_optimized.py --max_images 100 --epochs 15 --batch_size 8
```
- Fastest way to test if everything works
- Accuracy: 65-75%
- Good for testing

### Recommended Training (30-40 minutes)
```bash
python train_optimized.py --max_images 156 --epochs 30 --batch_size 8
```
- **Recommended for your dataset**
- Uses all 156 real images
- Accuracy: 80-88%
- Best balance of time/quality

### Maximum Quality (50-70 minutes)
```bash
python train_optimized.py --max_images 156 --epochs 50 --batch_size 8
```
- Longer training for better accuracy
- Accuracy: 85-92%
- Best if you have time

### If You Get Memory Errors
```bash
python train_optimized.py --max_images 156 --epochs 30 --batch_size 4
```
- Reduces memory usage
- Slightly slower but safer

---

## 📊 What Happens During Training

1. **Data Loading**: Reads images from dataset folders
2. **Preprocessing**: Resizes to 256x256, normalizes
3. **Model Building**: Creates CNN architecture
4. **Training**: 30-50 epochs with progress bar
5. **Validation**: Tests on 15% of data each epoch
6. **Saving**: Saves best model automatically

**What You'll See:**
```
Epoch 1/30
[====================] - 90s - loss: 0.45 - accuracy: 0.78 - val_loss: 0.35 - val_accuracy: 0.85
RAM: 62.3% | Available: 5.8GB
```

---

## ✅ After Training Completes

### Files Created:
- `model.h5` - Your trained model
- `best_model_optimized.h5` - Best checkpoint
- `training_history_optimized.png` - Training graphs
- `confusion_matrix_optimized.png` - Performance metrics

### Test the Model:

**Start Backend:**
```bash
python app.py
```
Opens at http://localhost:5000

**Start Frontend** (new terminal):
```bash
cd frontend
npm run dev
```
Opens at http://localhost:3000

**Or use launcher:**
```bash
start.bat
```

### Test API:
```bash
python test_api.py
```

---

## 🛠️ Troubleshooting

### "No module named 'pillow_heif'"
```bash
pip install pillow-heif
```

### "No module named 'psutil'"
```bash
pip install psutil
```

### "Out of Memory" Error
- Close other applications
- Use smaller batch size: `--batch_size 4`
- Use fewer images: `--max_images 100`

### Training is Slow
- **This is normal for CPU!**
- Your i7-6820HQ: ~2-3 minutes per epoch
- Total time: 30-90 minutes is expected
- You can use your computer during training (will be slow)

### Low Accuracy (<70%)
- Train longer: increase `--epochs 50`
- Use all images: `--max_images 156`
- Check dataset quality (real vs AI)

---

## 📈 Expected Performance

| Images | Epochs | Batch | Time    | Accuracy |
|--------|--------|-------|---------|----------|
| 100    | 15     | 8     | 20 min  | 65-75%   |
| 156    | 30     | 8     | 35 min  | 80-88%   |
| 156    | 50     | 8     | 60 min  | 85-92%   |

---

## 💡 Pro Tips

1. **Start with quick test first**
   - Make sure everything works
   - Then do full training

2. **Close applications before training**
   - Chrome/Edge can use 2-4GB RAM
   - Free up memory for training

3. **Don't interrupt training**
   - Let it complete
   - Model saves automatically

4. **Monitor resources**
   - Watch Task Manager
   - CPU: 80-100% is normal
   - RAM: Should stay below 14GB

5. **Training overnight**
   - If it takes too long
   - Start before sleep
   - Will complete automatically

---

## 🎯 Quick Command Reference

```bash
# Prepare (run once)
PREPARE_AND_TRAIN.bat

# Train (choose one)
python train_optimized.py --max_images 156 --epochs 30 --batch_size 8

# Test model
python app.py

# Launch full app
start.bat
```

---

## ✨ You're Ready!

1. Run `PREPARE_AND_TRAIN.bat`
2. Wait for dataset preparation
3. Copy and run the recommended training command
4. Wait for training to complete
5. Test with `python app.py`

**Good luck! 🚀**
