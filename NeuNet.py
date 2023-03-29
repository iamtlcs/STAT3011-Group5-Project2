from sklearn.model_selection import KFold
import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


class TripleTowersModel(keras.Model):
    def __init__(self, n_teams, n_confederations, n_tournaments, n_cities, n_countries, embedding_size=200, dense_size=128, embed_reg=1, dense_reg=1, fc_reg=1, **kwargs):
        super(TripleTowersModel, self).__init__(**kwargs)
        self.n_teams = n_teams
        self.n_confederations = n_confederations
        self.n_tournaments = n_tournaments
        self.n_cities = n_cities
        self.n_countries = n_countries
        
        ## Embedding layers
        self.home_embedding = layers.Embedding(
            self.n_teams,
            embedding_size,
            keras.initializers.he_normal(seed=None),
            embeddings_regularizer=keras.regularizers.l2(embed_reg),
        )
        self.away_embedding = layers.Embedding(
            self.n_teams,
            embedding_size,
            keras.initializers.he_normal(seed=None),
            embeddings_regularizer=keras.regularizers.l2(embed_reg),
        )
        self.home_confederations_embedding = layers.Embedding(
            self.n_confederations,
            embedding_size,
            keras.initializers.he_normal(seed=None),
            embeddings_regularizer=keras.regularizers.l2(embed_reg),
        )
        self.away_confederations_embedding = layers.Embedding(
            self.n_confederations,
            embedding_size,
            keras.initializers.he_normal(seed=None),
            embeddings_regularizer=keras.regularizers.l2(embed_reg),
        )
        self.tournaments_embedding = layers.Embedding(
            self.n_tournaments,
            embedding_size,
            keras.initializers.he_normal(seed=None),
            embeddings_regularizer=keras.regularizers.l2(embed_reg),
        )
        self.cities_embedding = layers.Embedding(
            self.n_cities,
            embedding_size,
            keras.initializers.he_normal(seed=None),
            embeddings_regularizer=keras.regularizers.l2(embed_reg),
        )
        self.countries_embedding = layers.Embedding(
            self.n_countries,
            embedding_size,
            keras.initializers.he_normal(seed=None),
            embeddings_regularizer=keras.regularizers.l2(embed_reg),
        )
        
        ##  Mapping layers
        self.home_dense = layers.Dense(dense_size, name='home_dense', activation='relu', 
                                       bias_regularizer=keras.regularizers.L2(dense_reg))
        self.away_dense = layers.Dense(dense_size, name='away_dense', activation='relu', 
                                       bias_regularizer=keras.regularizers.L2(dense_reg))
        self.info_dense = layers.Dense(dense_size, name='info_dense', activation='relu', 
                                       bias_regularizer=keras.regularizers.L2(dense_reg))
        self.joint_dense = layers.Dense(dense_size, name='joint_dense', activation='relu', 
                                       bias_regularizer=keras.regularizers.L2(dense_reg))
        
        self.fc1 = layers.Dense(dense_size, name='fc1', activation='relu', bias_regularizer=keras.regularizers.L2(fc_reg))
        self.fc2 = layers.Dense(dense_size, name='fc2', activation='relu', bias_regularizer=keras.regularizers.L2(fc_reg))
        self.out_dense = layers.Dense(1, name='Prediction', activation='sigmoid')

    def call(self, inputs):
        ## cate/cont data
        info_cont_feat, info_cate_feat, home_cont_feat, home_cate_feat, away_cont_feat, away_cate_feat = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]

        ## embedding
        home_vec = self.home_embedding(home_cate_feat[:,0])
        home_conf_vec = self.home_confederations_embedding(home_cate_feat[:,1])
        away_vec = self.away_embedding(away_cate_feat[:,0])
        away_conf_vec = self.away_confederations_embedding(away_cate_feat[:,1])
        tour_vec = self.tournaments_embedding(info_cate_feat[:,0])
        city_vec = self.cities_embedding(info_cate_feat[:,1])
        cont_vec = self.countries_embedding(info_cate_feat[:,2])

        ## dense mapping
        home_all_vec = layers.Concatenate()([home_cont_feat, home_vec, home_conf_vec])
        away_all_vec = layers.Concatenate()([away_cont_feat, away_vec, away_conf_vec])
        info_all_vec = layers.Concatenate()([info_cont_feat, tour_vec, city_vec, cont_vec])

        home_dense_vec = self.home_dense(home_all_vec)
        away_dense_vec = self.away_dense(away_all_vec)
        info_dense_vec = self.info_dense(info_all_vec)

        ## joint dense
        joint_vec = layers.Concatenate()([home_dense_vec, away_dense_vec, info_dense_vec])
        fc1_vec = self.joint_dense(joint_vec)
        fc2_vec = self.fc1(fc1_vec)
        out = self.out_dense(fc2_vec)
        return out
    
class TripleTowersModel_GridSearchKFoldCV(object):
    def __init__(self, n_teams, n_confederations, n_tournaments, n_cities, n_countries, 
                 cv=5, 
				 patiences=[3, 5, 7],
                 embed_regs=[1e-3, 1e-2, 1e-1], 
                 dense_regs=[1e-3, 1e-2, 1e-1], 
                 fc_regs=[1e-3, 1e-2, 1e-1],
                 embedding_sizes=[150, 200, 250, 300], 
                 dense_sizes=[150, 200, 250, 300],
                 lrs=[1e-4, 1e-3, 1e-2], batches=[64,128,256]):
        self.n_teams = n_teams
        self.n_confederations = n_confederations
        self.n_tournaments = n_tournaments
        self.n_cities = n_cities
        self.n_countries = n_countries
        self.cv = cv
        self.patiences = patiences
        self.embed_regs = embed_regs
        self.dense_regs = dense_regs
        self.fc_regs = fc_regs
        self.embedding_sizes = embedding_sizes
        self.dense_sizes = dense_sizes
        self.lrs = lrs
        self.batches = batches
        self.best_model = {}
        self.cv_result = {'embedding_size': [], 'dense_size': [], 'patience': [],
                          'embed_reg': [], 'dense_reg': [], 'fc_reg': [], 
                          'lr': [], 'batch': [], 'train_auc': [], 'valid_auc': []}

    def grid_search(self, train_input, train_rating):
        ## generate all combinations
        loop = self.cv*len(self.patiences)*len(self.embed_regs)*len(self.dense_regs)*len(self.fc_regs)*len(self.embedding_sizes)*len(self.dense_sizes)*len(self.lrs)*len(self.batches)
        count = 1
        kf = KFold(n_splits=self.cv, shuffle=True)
        for (embedding_size, dense_size, patience, embed_reg, dense_reg, fc_reg, lr, batch) in itertools.product(self.embedding_sizes, self.dense_sizes, self.patiences, self.embed_regs, self.dense_regs, self.fc_regs, self.lrs, self.batches):
            train_auc_tmp, valid_auc_tmp = 0., 0.
            for train_index, valid_index in kf.split(train_input[1]):
                print(f"Progress: {count} / {loop}")
                count = count + 1
                
                # produce training/validation sets
                train_input_cv = []
                valid_input_cv = []
                for i in range(6):
                    train_input_cv.append(train_input[i][train_index])
                train_rating_cv = train_rating[train_index]
                for i in range(6):
                    valid_input_cv.append(train_input[i][valid_index])
                valid_rating_cv = train_rating[valid_index]
                
                # fit the model based on CV data
                model = TripleTowersModel(self.n_teams, self.n_confederations, self.n_tournaments, self.n_cities, self.n_countries, 
                                          embedding_size=embedding_size, dense_size=dense_size,
                                          embed_reg=embed_reg, dense_reg=dense_reg, fc_reg=fc_reg)
				
                metrics = [keras.metrics.AUC(name='auc')]

                model.compile(optimizer=keras.optimizers.Adam(lr), loss=tf.keras.losses.BinaryCrossentropy(), metrics=metrics)

                callbacks = [keras.callbacks.EarlyStopping(monitor='val_auc', min_delta=0, patience=patience, verbose=1, mode='max', restore_best_weights=False)]

                history = model.fit(x=train_input_cv, y=train_rating_cv, batch_size=batch, epochs=100, verbose=1, callbacks=callbacks, validation_data=(valid_input_cv, valid_rating_cv))

                train_auc_tmp_cv = history.history["auc"][-1]
                valid_auc_tmp_cv = history.history["val_auc"][-1]
                train_auc_tmp = train_auc_tmp + train_auc_tmp_cv / self.cv
                valid_auc_tmp = valid_auc_tmp + valid_auc_tmp_cv / self.cv
                print(f'{self.cv}-Fold CV for embedding size: {embedding_size}; dense size: {dense_size}; patience: {patience}; embed reg: {embed_reg}; dense reg: {dense_reg}; fc reg: {fc_reg}; learning rate: {lr}; batch size: {batch}; train_auc: {train_auc_tmp_cv}, valid_auc: {valid_auc_tmp_cv}')
            self.cv_result['embedding_size'].append(embedding_size)
            self.cv_result['dense_size'].append(dense_size)
            self.cv_result['patience'].append(patience)
            self.cv_result['embed_reg'].append(embed_reg)
            self.cv_result['dense_reg'].append(dense_reg)
            self.cv_result['fc_reg'].append(fc_reg)
            self.cv_result['lr'].append(lr)
            self.cv_result['batch'].append(batch)
            self.cv_result['train_auc'].append(train_auc_tmp)
            self.cv_result['valid_auc'].append(valid_auc_tmp)
        self.cv_result = pd.DataFrame.from_dict(self.cv_result)
        best_ind = self.cv_result['valid_auc'].argmin()
        self.best_model = self.cv_result.loc[best_ind]
        
    def plot_grid(self, data_source='valid'):
        sns.set_theme()
        if data_source == 'train':
            cv_pivot = self.cv_result.pivot("embedding_size", "dense_size", "embed_reg", "dense_reg", "fc_reg",
                                            "lr", "batch", "train_auc")
        elif data_source == 'valid':
            cv_pivot = self.cv_result.pivot("embedding_size", "dense_size", "embed_reg", "dense_reg", "fc_reg",
                                            "lr", "batch", "valid_auc")
        else:
            raise ValueError('data_source must be train or valid!')
        sns.heatmap(cv_pivot, annot=True, fmt=".3f", linewidths=.5, cmap="YlGnBu")
        plt.show()