

class ValueEstimator():
    """
    Value Function approximator. 
    Scaler must implement .transform function transforming array of states to array of scaled states
    Featurizer must implement .transform function transforming array of scaled states to array of featurized states
    Model must implement sklearn .predict and .partial_fit functions 
    """
    
    def __init__(self, scaler, featurizer, model):
        self.scaler = scaler
        self.featurizer = featurizer
        self.model = model

    def featurize_state(self, state):
        """
        Returns the featurized representation for a state.
        """
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]
    
    def value(self, s):
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