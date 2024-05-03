CUDA_VISIBLE_DEVICES=0 python3 hyperparameter_optimization.py --train /users/ug21/atharvatambat/AML-project/CS726-OutlierOddballs/out_of_domain_unlabeled_data_Supplementary_Material/iclr_supp/Experiments/Histopathology/embeddings/cifar100_train_embeddings.pkl \
--train_unlabeled /users/ug21/atharvatambat/AML-project/CS726-OutlierOddballs/out_of_domain_unlabeled_data_Supplementary_Material/iclr_supp/Experiments/Histopathology/embeddings/cifar5m_train_embeddings.pkl \
--test /users/ug21/atharvatambat/AML-project/CS726-OutlierOddballs/out_of_domain_unlabeled_data_Supplementary_Material/iclr_supp/Experiments/Histopathology/embeddings/cifar100_test_embeddings.pkl \
--output /users/ug21/atharvatambat/AML-project/CS726-OutlierOddballs/out_of_domain_unlabeled_data_Supplementary_Material/iclr_supp/Experiments/Histopathology/cifar100_odnl_combined/ \
--current_dir "$PWD" \
--hpnum 1 \
--labeled_number 100 \
--unlabeled_number 7000 \
--use_scheduler True \
--use_weighted_loss False \
<<<<<<< HEAD
--same_dist_ul False \
--frac_random_labelled 0.5 \
--num_classes 100 \
--dataset cifar100
=======
--same_dist_ul False\
--num_classes 2
>>>>>>> 8bc9a5c47d2014226c971a69eefe1e88753f6f41
