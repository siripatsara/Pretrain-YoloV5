# YOLOv5 Segmentation Training with TensorBoard Integration

This enhanced training script provides comprehensive TensorBoard logging for monitoring YOLOv5 segmentation model training progress.

## Features Added

### 1. Loss Tracking
- **Box Regression Loss**: Tracks bounding box coordinate prediction accuracy
- **Objectness Loss**: Monitors object detection confidence
- **Classification Loss**: Tracks class prediction accuracy  
- **Segmentation Loss**: Monitors segmentation mask prediction quality

### 2. Training Metrics
- **Learning Rate**: Visualizes learning rate schedule
- **Training Progress**: Real-time loss monitoring per batch and epoch
- **Validation Metrics**: Precision, Recall, mAP@0.5, mAP@0.5:0.95

### 3. Model Analysis
- **Precision-Recall Curves**: Visual representation of model performance with mAP calculation
- **Accuracy Plots**: Training and validation accuracy over time
- **Parameter Histograms**: Weight and gradient distribution analysis (logged every 10 epochs)

### 4. Visual Monitoring
- **Training Images**: Sample training batches with annotations
- **Model Convergence**: All loss curves to monitor training stability

## Usage

### Basic Usage
```bash
python segment/train_segment_tensor.py --tensorboard --data data/coco128-seg.yaml --weights yolov5s-seg.pt
```

### Complete Example
```bash
python segment/train_segment_tensor.py \
    --data data/coco128-seg.yaml \
    --weights yolov5s-seg.pt \
    --img 640 \
    --batch-size 16 \
    --epochs 100 \
    --tensorboard \
    --name my_experiment \
    --device 0
```

### Using the Example Script
```bash
python segment/train_with_tensorboard_example.py
```

## Viewing TensorBoard Logs

1. **Start TensorBoard**:
   ```bash
   tensorboard --logdir runs/train-seg/[experiment_name]/tensorboard
   ```

2. **Open Browser**: Navigate to http://localhost:6006/

3. **Available Tabs**:
   - **Scalars**: Loss curves, metrics, learning rate
   - **Images**: Training samples and visualizations  
   - **Histograms**: Model parameters and gradients
   - **Graphs**: Model architecture (if available)

## TensorBoard Dashboard Sections

### Scalars Tab
- `Loss/Train_Box` - Box regression loss during training
- `Loss/Train_Objectness` - Objectness loss during training
- `Loss/Train_Classification` - Classification loss during training  
- `Loss/Train_Segmentation` - Segmentation loss during training
- `Loss/Train_Total` - Total training loss
- `Loss/Epoch_*` - Average losses per epoch
- `Loss/Val_*` - Validation losses
- `Metrics/Precision` - Model precision
- `Metrics/Recall` - Model recall
- `Metrics/mAP@0.5` - Mean Average Precision at IoU=0.5
- `Metrics/mAP@0.5:0.95` - Mean Average Precision at IoU=0.5:0.95
- `Metrics/Accuracy` - Model accuracy
- `Metrics/Fitness` - Overall model fitness score
- `Learning_Rate/LR` - Learning rate schedule
- `Precision-Recall/mAP@0.5_Calculated` - Calculated mAP from PR curve

### Images Tab
- `Training/Images` - Sample training images from first batch of each epoch

### Histograms Tab
- `Parameters/*` - Model weight distributions
- `Gradients/*` - Gradient distributions (useful for debugging training)

### Additional Features Tab
- `Precision-Recall/Curve` - Interactive PR curves with mAP visualization

## Command Line Arguments

New TensorBoard-specific argument:
- `--tensorboard`: Enable TensorBoard logging (default: False)

## Requirements

Make sure you have the required packages:
```bash
pip install tensorboard matplotlib scikit-learn
```

## Tips for Effective Monitoring

1. **Start TensorBoard Early**: Launch TensorBoard before starting training to see real-time updates
2. **Monitor Convergence**: Watch for loss curves plateauing or increasing (potential overfitting)
3. **Learning Rate**: Ensure learning rate decreases appropriately over time
4. **Validation Metrics**: Monitor mAP and precision/recall for actual performance
5. **Parameter Analysis**: Use histograms to detect vanishing/exploding gradients

## Troubleshooting

### Common Issues:
1. **TensorBoard not updating**: Refresh browser or restart TensorBoard
2. **Port conflicts**: Use `--port` flag with TensorBoard: `tensorboard --logdir path --port 6007`
3. **Memory issues**: Reduce `--batch-size` if running out of GPU memory
4. **Slow logging**: Parameter histograms are logged every 10 epochs to reduce overhead

### Performance Notes:
- TensorBoard logging adds minimal overhead (~1-2% training time)
- Parameter histogram logging is limited to every 10 epochs for performance
- Image logging is limited to first batch of each epoch

## Example Output Structure
```
runs/train-seg/
└── experiment_name/
    ├── tensorboard/          # TensorBoard log files
    │   ├── events.out.tfevents.*
    │   └── ...
    ├── weights/
    │   ├── best.pt
    │   └── last.pt
    ├── results.csv
    └── ...
```

This enhanced training script provides comprehensive monitoring capabilities to help you understand your model's training dynamics and achieve better results.
