import numpy as np
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist,squareform

random_matrix = np.array([0,4,657,34,6557])





class SVDBaseRecSys : 
    
    
    """
    SVDBaseRecSys based on implementation of latent factor of SVD 
    
    
    >>user_rating_matrix = np.array([
                [1,np.nan,3,np.nan,np.nan,5,np.nan,np.nan,5,np.nan,4,np.nan],
                [np.nan,np.nan,5,4,np.nan,np.nan,4,np.nan,np.nan,2,1,3],
                [2,4,np.nan,1,2,np.nan,3,np.nan,4,3,5,np.nan],
                [np.nan,2,4,np.nan,5,np.nan,np.nan,4,np.nan,np.nan,2,np.nan],
                [np.nan,np.nan,4,3,4,2,np.nan,np.nan,np.nan,np.nan,2,5],
                [1,np.nan,3,np.nan,3,np.nan,np.nan,2,np.nan,np.nan,4,np.nan]]).T
    >> model = SVDBaseRecSys()
    >> model.fit(user_rating_matrix)

    
    
    """
    def __init__(self,missing_imputation='negative',n_factor=10,n_epochs=100,lr=0.005,regularization_terms=0.02,random_state=42) -> None:
        """Initialization Process

        Args:
            similarity_method (str, optional): Similarity Method . Defaults to 'cosine'.
            missing_imputation (str, optional): _description_. Defaults to 'negative'.
            n_neighbour (int, optional): _description_. Defaults to 5.
        """
        self.missing_imputation = missing_imputation
        self.n_factor  = n_factor
        self.n_epochs = n_epochs 
        self.regularization_terms = regularization_terms # as default in yehuda koren 
        self.lr = lr
        self.loss = 0 
        self.state = np.random.RandomState(random_state)

    def convert_utility_matrix(self,utility_matrix) :
        """Convert utility matrix with structure of n_user x n_items with value ratings into array with 3 column structure 
            Col 1 -> user_id 
            Col 2 -> item_id 
            Col 3 -> ratings 

        Args:
            utility_matrix (_type_): _description_
        """
        data = pd.DataFrame(utility_matrix)
        data['users'] = [x for x in range(self.n_users)]
        utm = data.melt(id_vars=['users'])
        utm = utm.rename(columns={'variable':'items','value':'ratings'})
        utm['items'] = utm['items'].astype('int')
        utm['users'] = utm['users'].astype('int')
        utm['ratings'] = utm['ratings'].astype('float')
        utility_matrix = utm.to_numpy()
        return utility_matrix
        
        
    def preprocess_user_rating_matrix(self,user_rating_matrix) :
        """
        Missing Values Imputation of User Rating Matrix using Mean of average item ratings ( Based on Charu C Aggarwal on Dimensionality Reduction Neighbourhood)
        However According to Original Paper it said that the missing values treated as true Negative rating , some simple explanation may be in scala 0 5 ratings negative mean 0 ratings
        
        Args:
            user_rating_matrix (_type_): _description_

        Returns:
            _type_: _description_
        """
        #save missing location for prediction purpose 
        self.missing_location = np.argwhere(np.isnan(user_rating_matrix))
        for col in range(user_rating_matrix.shape[0]) : 
            mean = np.nanmean(user_rating_matrix[col])
            user_rating_matrix[col][np.isnan(user_rating_matrix[col])] = 0


        return user_rating_matrix 

    
    

        

    
    def fit(self,X) : 
        self.original_utility_matrix = X 
        self.user_rating_matrix = self.preprocess_user_rating_matrix(X) 
        self.n_users,self.n_items = X.shape
        self.user_rating_matrix = self.convert_utility_matrix(self.user_rating_matrix)
        
        
        
        #initialize user bias and items 
        self.user_bias = np.zeros(shape=(self.n_users,1))
        self.item_bias = np.zeros(shape=(self.n_items,1))

        #initialize factors (SVD Component)
        self.u_factor = self.state.normal(size=(self.n_users,self.n_factor))
        self.v_factor = self.state.normal(size=(self.n_items,self.n_factor))
        
        #find global mean of the data 
        self.global_mean = np.nanmean(X)
        
        #utility matrix len 
        loop_length = len(self.user_rating_matrix)
        #epoch or loops 
        for epoch in range(self.n_epochs) : 
            print(f'epochs :{epoch+1} / {self.n_epochs}')
            for idx in range(loop_length) : 
                
                user_id =  int(self.user_rating_matrix[idx][0])
                item_id =  int(self.user_rating_matrix[idx][1])
                rating =  self.user_rating_matrix[idx][2]

                
                if rating == np.nan : 
                    continue 
                
                user_bias = self.user_bias[user_id]
                
                item_bias = self.item_bias[item_id]
                
                #get latent factor of user_id 

                u_factor_user = self.u_factor[user_id]
                # get latent factor of user_
                v_factor_item = self.v_factor[item_id]
                
                # dot product between <u_factor_user,v_factor_item> 
                dot = np.dot(u_factor_user,v_factor_item.T)

                #calculate prediction of ratings 
                r_hat = self.global_mean + user_bias + item_bias + dot 

                #calculate error , using MSE 
                current_loss = (rating - r_hat)

                self.loss += current_loss 
                
                #update user bias , bu ← bu + γ · (eui − λ4 · bu)
                delta_user_bias =  self.lr*(current_loss - self.regularization_terms*user_bias)

                self.user_bias[user_id] += delta_user_bias
                 
                #update item bias bi ← bi + γ · (eui − λ4 · bi)
                delta_item_bias =   self.lr*(current_loss - self.regularization_terms*item_bias)
                self.item_bias[item_id] += delta_item_bias
                
                #update latent factor of user  pu ← pu + γ · (eui · qi − λ4 · pu)
                delta_u_factor =   self.lr*(current_loss*v_factor_item - self.regularization_terms*u_factor_user)

                self.u_factor[user_id] += delta_u_factor
                
                #update latent factor of item qi ← qi + γ · (eui · pu − λ4 · qi)
                delta_v_factor =   self.lr*(current_loss*u_factor_user - self.regularization_terms*v_factor_item)

                self.v_factor[item_id] +=delta_v_factor

                

            
            
        
    def _predict(self,user_x,item_i) : 
        """_summary_

        Args:
            user_x (_type_): _description_
            item_i (_type_): _description_

        Returns:
            _type_: _description_
        """
        
        
        

        
        return None 
            
        
        
        
    def get_recommendation(self,user_idx,recommend_only_missing=True,top_k=5) : 
        """Recommend best top K item for user=user_idx
            Approach -> repredict missing rating only -> sort best on the highest-k (could be set)
        Args:
            user_idx (_type_): user_idx to recommend

        Returns:
            _type_: _description_
        """
        #predict only missing value of the data 
        #finding missing_value on spesific user_idx
        missing_item_idx = []
        #missing location has shape of mxn (similar of user rating matrix) m -> user_idx and n-> item_index
        for missing_loc in self.missing_location : 
            if missing_loc[0]==user_idx : 
                missing_item_idx.append(missing_loc[1])
            else : 
                continue 
        #call user_rating_matrix = 
        user_idx_rating_matrix = self.user_rating_matrix[user_idx]
        
        #refill again the missing ones 
        for idx in missing_item_idx : 
            user_idx_rating_matrix[idx]= self._predict(user_x=user_idx,item_i=idx)
        #with assumption that rated item will not be recommended again 
        
        recommendation  = {}
        if recommend_only_missing : 
            missing_ratings = user_idx_rating_matrix[missing_item_idx]
            rank = np.argsort(missing_ratings)[::-1][:top_k]
            sorted_ratings = missing_ratings[rank]
        
        
            for x,y in zip(rank,sorted_ratings) : 
                recommendation[f'Item ID : {x} ']= y
        else : 
            rank = np.argsort(user_idx_rating_matrix)[::-1][:top_k]
            sorted_ratings = user_idx_rating_matrix[rank]
            for x,y in zip(rank,sorted_ratings) : 
                recommendation[f'Item ID : {x} ']= y
                
        return recommendation
    
    




example_URM = np.array([
            [1,np.nan,3,np.nan,np.nan,5,np.nan,np.nan,5,np.nan,4,np.nan],
            [np.nan,np.nan,5,4,np.nan,np.nan,4,np.nan,np.nan,2,1,3],
            [2,4,np.nan,1,2,np.nan,3,np.nan,4,3,5,np.nan],
            [np.nan,2,4,np.nan,5,np.nan,np.nan,4,np.nan,np.nan,2,np.nan],
            [np.nan,np.nan,4,3,4,2,np.nan,np.nan,np.nan,np.nan,2,5],
            [1,np.nan,3,np.nan,3,np.nan,np.nan,2,np.nan,np.nan,4,np.nan]]).T


recsys = SVDBaseRecSys()

recsys.fit(example_URM)

print('bias',recsys.v_factor)
# print(recsys._predict(user_x=0,item_i=0))

# print(recsys.get_recommendation(user_idx=0))