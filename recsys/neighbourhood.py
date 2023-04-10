import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist,squareform

random_matrix = np.array([0,4,657,34,6557])





class NeigbourhoodRecSys : 
    
    
    """
    User Based NeigbourhoodRecSys : Explicit Ratings
    Requirements : 
    1.User Rating Matrix
    2.Similarity Method (Cosine,Pearson)
    3.Imputation Method for Sparse Matrix 
    4.Prediction Function -> accomodate bias,etc
    5.Preprocessing Function --> substracting mean -> 
    
    
    """
    def __init__(self,similarity_method='cosine',missing_imputation='negative',n_neighbour=5) -> None:
        self.similarity_method= similarity_method
        self.missing_imputation = missing_imputation
        self.n_neighbour = n_neighbour
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
        
    def find_similar_neighbour(self,user_x) : 
        
        neighbour = np.argsort(self.similarity_matrix[user_x])[::-1][:self.n_neighbour]
        return neighbour
    def calculate_similarity(self,a,b) : 
        if self.similarity_method == 'cosine' : 
             return cosine_similarity(a.reshape(1, -1),b.reshape(1, -1))[0]
        elif self.similarity_method == 'pearson' : 
            NotImplementedError('Method has not added') 
        else : 
            raise NotImplementedError('Method has not added')
    def create_similarity_matrix(self,user_matrix) : 
        if self.similarity_method == 'cosine' : 
            return squareform(pdist(user_matrix, metric='cosine')) 
        elif self.similarity_method == 'pearson' : 
            pass 
        else : 
            raise NotImplementedError('Method has not added') 
    
    def create_baseline_prediction(self,user_x,item_i) : 
        """

        Args:
            user_x (_type_): _description_
            item_i (_type_): _description_
            
            
        $$with $$
        $$ b_{xi} = \mu +  b_x + b_i $$

        $$ b_{xi} = baselines\, estimation\, for\,user\,on\,item\,i  $$
        $$ \mu = overall\, mean\, rating  $$
        $$ b_x  = rating\, deviation\,for\,user\,x  $$
        $$ b_i  = rating\, deviation\,for\,item\,i  $$
        """
        ##compute overall mean \mu
        mu = self.user_rating_matrix.mean()
        ## compute rating deviation of user x 
        bx = self.user_rating_matrix[user_x].mean() - mu
        ## compute rating deviation of user x 
        bi = self.user_rating_matrix.mean(axis=1)[item_i] - mu 
        
        baseline = mu + bx + bi
        
        return baseline 
        

    
    def fit(self,X) : 
        self.user_rating_matrix = self.preprocess_user_rating_matrix(X) 
        self.similarity_matrix = self.create_similarity_matrix(self.user_rating_matrix)
    def _predict(self,user_x,item_i) : 
        """
        $$\hat{r_{xi}} = b_{xi} + \frac{\Sigma_{j \subset N (i;x)} s_{ij}.(r_{xj}-b_{xj}) }{\Sigma_{j \subset N (i;x)} s_{ij}}$$

        $$with $$
        $$ b_{xi} = \mu +  b_x + b_i $$

        $$ b_{xi} = baselines\, estimation\, for\,user\,on\,item\,i  $$
        $$ \mu = overall\, mean\, rating  $$
        $$ b_x  = rating\, deviation\,for\,user\,x  $$
        $$ b_i  = rating\, deviation\,for\,item\,i  $$
        """ 
        # check if user rating matrix exists -> if exists -> has been fitted before 
        if self.user_rating_matrix is None : 
            raise ValueError('You cant Predict Now, Fit first!')
        #finding baselines estimation 
        baselines_xi = self.create_baseline_prediction(user_x=user_x,item_i=item_i)

        #find k most similar neighbour 
        similar_neighbour_idxs = self.find_similar_neighbour(user_x=user_x)
        
        nominator = []
        denominator = []
        for neigbour_idx in similar_neighbour_idxs : 
            #calculate similarity between user x and neigbour_idx
            user_x_rating_vec = self.user_rating_matrix[user_x]
            neighbour_rating_vec = self.user_rating_matrix[neigbour_idx] 
            similarity_x_neighbour = self.calculate_similarity(a=user_x_rating_vec,b=neighbour_rating_vec)
            denominator.append(similarity_x_neighbour)
            #find baseline rating
            baseline = self.create_baseline_prediction(user_x=neigbour_idx,item_i=item_i)
            #calculate the adjusted rating of neighbor_idx on item_i 
            unadjusted_rating = self.user_rating_matrix[neigbour_idx][item_i]
            #adjusted rating 
            adjusted_rating = unadjusted_rating - baseline 
            
            #multiply similarity and adjusted_rating 
            mul = similarity_x_neighbour * adjusted_rating 
            nominator.append(mul)
        rating_x_i = baselines_xi + (np.sum(nominator) / np.sum(denominator))
        
        return rating_x_i
            
        
        
        
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
    
    



if __name__ == '__main__' : 
    example_URM = np.array([
                [1,np.nan,3,np.nan,np.nan,5,np.nan,np.nan,5,np.nan,4,np.nan],
                [np.nan,np.nan,5,4,np.nan,np.nan,4,np.nan,np.nan,2,1,3],
                [2,4,np.nan,1,2,np.nan,3,np.nan,4,3,5,np.nan],
                [np.nan,2,4,np.nan,5,np.nan,np.nan,4,np.nan,np.nan,2,np.nan],
                [np.nan,np.nan,4,3,4,2,np.nan,np.nan,np.nan,np.nan,2,5],
                [1,np.nan,3,np.nan,3,np.nan,np.nan,2,np.nan,np.nan,4,np.nan]]).T


    recsys = NeigbourhoodRecSys()

    recsys.fit(example_URM)

    print(recsys._predict(user_x=0,item_i=0))
    
    print(recsys.get_recommendation(user_idx=0))