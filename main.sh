# export MODELNAME='Thyroid_PF'
# export ARCH="densnet"
# export IMAGEPATH="./dataset"
# export TRAIN="CXP_train_age"
# export VALID="CXP_valid_age"
# export TEST="CXP_female_age" # CXP_test_age, CXP_male_age

# export MODELNAME='Thyroid_PM'
# export ARCH="densnet"
# export IMAGEPATH="./dataset"
# export TRAIN="CXP_train_age"
# export VALID="CXP_valid_age"
# export TEST="CXP_female_age" # CXP_test_age, CXP_male_age

# export MODELNAME='Thyroid_TC'
# export ARCH="densnet"
# export IMAGEPATH="./dataset"
# export TRAIN="CXP_train_age"
# export VALID="CXP_valid_age"
# export TEST="CXP_female_age" # CXP_test_age, CXP_male_age

# export MODELNAME='CXP_Age'
# export ARCH="densnet"
# export IMAGEPATH="./dataset"
# export TRAIN="CXP_train_age"
# export VALID="CXP_valid_age"
# export TEST="CXP_female_age" # CXP_test_age, CXP_male_age
# export BATCH=64

# export MODELNAME='CXP_Race'
# export ARCH="densnet"
# export IMAGEPATH="/export/home/daifang/CXP/dataset"
# export TRAIN="CXP_train_race"
# export VALID="CXP_valid_race"
# export TEST="CXP_female_race" # CXP_test_race, CXP_male_age
# export BATCH=124

# export MODELNAME='ISIC2019_Sex'
# export ARCH="efficientnet"
# export IMAGEPATH="./dataset"
# export TRAIN="ISIC_2019_Training_sex"
# export VALID="ISIC_2019_valid"
# export TEST="ISIC_2019_Test" # CXP_test_race, CXP_male_age
# export BATCH=114

# export MODELNAME='ISIC2019_Age'
# export ARCH="efficientnet"
# export IMAGEPATH="./dataset"
# export TRAIN="ISIC_2019_Training_age"
# export VALID="ISIC_2019_valid"
# export TEST="ISIC_2019_Test" # CXP_test_race, CXP_male_age
# export BATCH=114

python main_train.py \
    --modelname $MODELNAME   \
    --architecture $ARCH   \
    --imagepath $IMAGEPATH   \
    --train_data $TRAIN   \
    --valid_data $VALID   \
    --test_data $TEST  \
    --learning_rat 0.0001  \
    --batch_size $BATCH   \
    --num_epochs 1000 

