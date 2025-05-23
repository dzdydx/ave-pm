
# !/bin/bash


# select baselines and run experiments
BASELINE="AVEL" # [AVEL, CMBS, LAVISH, CPSP]
EVALUATE="False" # [True, False]

if [ "$EVALUATE" = "False" ]; then
    if [ "$BASELINE" = "CPSP" ]; then
        echo "Running CPSP baseline"
        cd ./CPSP
        python main.py --config config/cpsp.yaml

    elif [ "$BASELINE" = "AVEL" ]; then
        echo "Running AVEL baseline"
        cd ./CPSP
        python main.py --config /data1/cy/nips/CPSP/config/test.yaml

    elif [ "$BASELINE" = "CMBS" ]; then
        echo "Running CMBS baseline"
        cd ./CMBS
        bash supv_train.sh
    elif [ "$BASELINE" = "LAVISH" ]; then
        echo "Running LAVISH baseline"
        cd ./LAVISH
        bash train.sh
    fi
else
    if [ "$BASELINE" = "CPSP" ]; then
        echo "Evaluating CPSP baseline"
        cd ./CPSP
        python main.py --config config/cpsp.yaml
    elif [ "$BASELINE" = "AVEL" ]; then
        echo "Evaluating AVEL baseline"
        cd ./CPSP
        python main.py --config config/avel.yaml
    elif [ "$BASELINE" = "CMBS" ]; then
        echo "Evaluating CMBS baseline"
        cd ./CMBS
        bash supv_test.sh
    elif [ "$BASELINE" = "LAVISH" ]; then
        echo "Evaluating LAVISH baseline"
        cd ./LAVISH
        bash test.sh
    fi
fi



