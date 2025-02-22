CUDA_VISIBLE_DEVICES=4 python3 training.py --train /users/ug21/atharvatambat/AML-project/CS726-OutlierOddballs/out_of_domain_unlabeled_data_Supplementary_Material/iclr_supp/Experiments/Histopathology/embeddings/cifar10_train_embeddings.pkl \
--train_unlabeled /users/ug21/atharvatambat/AML-project/CS726-OutlierOddballs/out_of_domain_unlabeled_data_Supplementary_Material/iclr_supp/Experiments/Histopathology/embeddings/cifar5m_train_embeddings.pkl \
--test /users/ug21/atharvatambat/AML-project/CS726-OutlierOddballs/out_of_domain_unlabeled_data_Supplementary_Material/iclr_supp/Experiments/Histopathology/embeddings/cifar10_test_embeddings.pkl \
--output /users/ug21/atharvatambat/AML-project/CS726-OutlierOddballs/out_of_domain_unlabeled_data_Supplementary_Material/iclr_supp/Experiments/Histopathology/results/ \
--current_dir "$PWD" \
--hpnum 1 \
--labeled_number 1000 \
--unlabeled_number 0 \
--use_scheduler True \
--use_weighted_loss False \
--same_dist_ul False