# # bi-attention resnet-50 FINETUNE experiments
# echo "Cars196 experiments\n"
# python src/resnet50_trainer_bat.py --multigpu_config config/cars196/bat_training_baseline.yaml --model_config config/cars196/bat_model_baseline.yaml 
# 
# echo "\n"
# echo "\n"
# echo "\n"
# echo "\n"
# echo "\n"
# echo "Birds200 experiments\n"
# python src/resnet50_trainer_bat.py --multigpu_config config/birds200/bat_training_baseline.yaml --model_config config/birds200/bat_model_baseline.yaml 
# 
# echo "\n"
# echo "\n"
# echo "\n"
# echo "\n"
# echo "\n"
# echo "Aircrafts100 experiments\n"
# python src/resnet50_trainer_bat.py --multigpu_config config/aircrafts100/bat_training_baseline.yaml --model_config config/aircrafts100/bat_model_baseline.yaml 


# # bi-attention RETRAIN experiments
# echo "Cars196 experiments"
# python src/resnet50_trainer_bat_retrain.py --multigpu_config config/cars196/bat_training_baseline_retrain.yaml --model_config config/cars196/bat_model_baseline_retrain.yaml 
# 
# echo ""
# echo ""
# echo ""
# echo ""
# echo ""
# echo "Birds200 experiments"
# python src/resnet50_trainer_bat_retrain.py --multigpu_config config/birds200/bat_training_baseline_retrain.yaml --model_config config/birds200/bat_model_baseline_retrain.yaml 
# 
# echo ""
# echo ""
# echo ""
# echo ""
# echo ""
echo "Aircrafts100 experiments"
python src/resnet50_trainer_bat_retrain.py --multigpu_config config/aircrafts100/bat_training_baseline_retrain.yaml --model_config config/aircrafts100/bat_model_baseline_retrain.yaml 
