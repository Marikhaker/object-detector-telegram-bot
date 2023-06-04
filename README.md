# Video object detection using Ultralytics YOLOv8 and Telegram-bot
Run by starting bot.py. 

## Demo
https://github.com/Marikhaker/object-detector-telegram-bot/assets/55658081/a4ebcd25-5d04-46d5-b17e-c019e8bb791a

## Acrhitecture
<div align="center">
  <p>
     <img width="40%" src="https://github.com/Marikhaker/object-detector-telegram-bot/blob/main/training_summary/system%20architecture.jpg?raw=true">
  </p>
</div>

## Training results
Trained YOLOv8s for 22 epoches for 14 classes. Trained on part of Open-Images-v7 dataset: 15k for train, 1k for validate. Dataset downloaded with FiftyOne.
```
custom_14_cls = ["Clothing", "Tree", "Human face", "Footwear", "Window",
"Building", "Chair", "Fashion accessory",
"Food", "Glasses", "Sunglasses", "Vehicle", "Drink", "Animal"]
```

### Training curves
<div align="center">
  <p>
     <img width="70%" src="https://github.com/Marikhaker/object-detector-telegram-bot/blob/main/training_summary/results.png?raw=true">
  </p>
</div>

### Confusion matrix
<div align="center">
  <p>
     <img width="70%" src="https://github.com/Marikhaker/object-detector-telegram-bot/blob/main/training_summary/confusion_matrix_normalized.png?raw=true">
  </p>
</div>

### Classification report
<div align="center">
  <p>
    <img width="70%" src="https://github.com/Marikhaker/object-detector-telegram-bot/blob/main/training_summary/classification_report.jpg?raw=true">
  </p>
</div>
