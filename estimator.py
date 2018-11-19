class Estimator():
    """
    Value Function approximator. 
    """
    
    def __init__(self, scaler, featurizer):
        self.model = SGDRegressor(learning_rate="constant")
        # We need to call partial_fit once to initialize the model
        # or we get a NotFittedError when trying to make a prediction
        # This is quite hacky.
        self.scaler = scaler
        self.featurizer = featurizer
        self.model.partial_fit([self.featurize_state(s_arr[0,:])], [0])
    
    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]
    
    def predict(self, s):
        """
        Makes value function predictions.
        """
        features = self.featurize_state(s)
        return self.model.predict([features])[0]
    
    def update(self, s, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        features = self.featurize_state(s)
        self.model.partial_fit([features], y)