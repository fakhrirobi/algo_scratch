class SVDplus : 
    
    
    """
   SVDplus -> additional variable concern with implicit feedback on whether user rated spesific item or not 
    
    
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
    def __init__(self,missing_imputation='negative',n_factor=10,n_epochs=100,lr=0.005,regularization_terms=0.02,random_state=42,
                 impute_missing=False) -> None:
        """
        Initialization Process of SVD Based Model for Recommender System . Idea is pretty straightforward. Rating matrix could be decomposed into 3 separate matrix 
        and the model itself is trying to do the reverse -> build matrix random first -> optimize with SGD -> get the optimal matrix 

        Args:
            missing_imputation (str, optional): _description_. Defaults to 'negative'.
            n_factor (int, optional): _description_. Defaults to 10.
            n_epochs (int, optional): _description_. Defaults to 100.
            lr (float, optional): _description_. Defaults to 0.005.
            regularization_terms (float, optional): _description_. Defaults to 0.02.
            random_state (int, optional): _description_. Defaults to 42.
            impute_missing (bool, optional): _description_. Defaults to False.
        """
        self.missing_imputation = missing_imputation
        self.n_factor  = n_factor
        self.n_epochs = n_epochs 
        self.regularization_terms = regularization_terms # as default in yehuda koren 
        self.lr = lr
        self.loss = 0 
        self.state = np.random.RandomState(random_state)
        self.impute_missing = impute_missing 

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
            user_rating_matrix[col][np.isnan(user_rating_matrix[col])] = self.global_mean


        return user_rating_matrix 

    
    

        

    
    def fit(self,X) : 
        self.original_utility_matrix = X 
        self.global_mean = np.nanmean(X)
        self.n_users,self.n_items = X.shape
        if self.impute_missing : 
            
            self.user_rating_matrix = self.preprocess_user_rating_matrix(X) 
            self.user_rating_matrix = self.convert_utility_matrix(self.user_rating_matrix)
        
        
        self.user_rating_matrix = self.convert_utility_matrix(X)
        
        
        
        #initialize user bias and items 
        self.user_bias = np.zeros(shape=(self.n_users,1))
        self.item_bias = np.zeros(shape=(self.n_items,1))

        #initialize factors (SVD Component)
        self.u_factor = self.state.normal(size=(self.n_users,self.n_factor))
        self.v_factor = self.state.normal(size=(self.n_items,self.n_factor))
        
        #initialize y as implicit feedback where user who rates certain items as 1 otherwise zero 
        self.implicit_feedback = np.zeros(self.n_factor)
        
        #initialize yj as implicit feedback which value as 1 if user has rated it 
        self.y_j  = self.state.normal(size=(self.n_items,self.n_factor))
        #utility matrix len 
        loop_length = len(self.user_rating_matrix)
        #epoch or loops 
        for epoch in range(self.n_epochs) : 
            print(f'epochs :{epoch+1} / {self.n_epochs}')
            for idx in range(loop_length) : 
                
                user_id =  int(self.user_rating_matrix[idx][0])
                item_id =  int(self.user_rating_matrix[idx][1])
                rating =  self.user_rating_matrix[idx][2]
                print('rating',rating)
                
                if np.isnan(ratings) : 
                    continue 
                
                user_bias = self.user_bias[user_id]
                
                item_bias = self.item_bias[item_id]
                
                #get latent factor of user_id 

                u_factor_user = self.u_factor[user_id]
                # get latent factor of user_
                v_factor_item = self.v_factor[item_id]
                #update  implicit feedback 
                implicit_item_id = self.implicit_feedback[item_id]
                
                
                # dot product between <u_factor_user,v_factor_item+implicit feedback> 
                dot = np.dot(u_factor_user,v_factor_item.T)

                #calculate prediction of ratings 
                r_hat = self.global_mean + user_bias + item_bias + dot 

                #calculate error , using MSE + regularization terms 
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

                #update 
                

            
            
        
    def _predict(self,user_id,item_id) : 
        """_summary_

        Args:
            user_x (_type_): _description_
            item_i (_type_): _description_

        Returns:
            _type_: _description_
        """
        user_bias = self.user_bias[user_id]
        item_bias = self.item_bias[item_id]
        
        u_factor_user = self.u_factor[user_id]
        # get latent factor of user_
        v_factor_item = self.v_factor[item_id]
        
        # dot product between <u_factor_user,v_factor_item> 
        dot = np.dot(u_factor_user,v_factor_item.T)

        #calculate prediction of ratings 
        r_hat = self.global_mean + user_bias + item_bias + dot 
        
        

        
        return r_hat 
            
        
        
        
    def get_recommendation(self,user_id,recommend_only_missing=True,top_k=5) : 
        ratings = []
        highest_val = 0 
        list_of_recommendation = []
        for idx in range(len(self.user_rating_matrix)) : 
            recommendation = {'user': user_id}
            user = int(self.user_rating_matrix[idx][0])
            if user != user_id : 
                continue 
            item_id =  int(self.user_rating_matrix[idx][1])

            r_hat = self._predict(user_id=user,item_id=item_id)
            ratings.append(r_hat)
        ratings = np.array(ratings)
        rank = np.argsort(ratings)[::-1][:top_k]
        sorted_ratings = ratings[rank]
        for x,y in zip(rank,sorted_ratings) : 
            recommendation = {}
            recommendation[f'Item ID : {x} ']= y
            list_of_recommendation.append(recommendation)
        return list_of_recommendation
