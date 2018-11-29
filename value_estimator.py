

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

    def featurize_states(self, state):
        """
        Returns the featurized representation for an array of states. Takes in 2D array of states. Returns 2D array of features.
        """
        scaled = self.scaler.transform(state)
        featurized = self.featurizer.transform(scaled)
        return featurized
    
    def values(self, s):
        """
        Makes value function prediction on an array of states. Takes in a 2D array, returns 1D array list of values.
        """
        features = self.featurize_states(s)
        return self.model.predict(features)

    def value(self, s):
        """
        Makes value function prediction on a single state. Takes in a 2D array (1-length in axis 0) [[state...]]
        """
        features = self.featurize_states(s)
        return self.model.predict(features)[0]
    
    def update(self, s, y):
        """
        Updates the estimator parameters for a given state and action towards
        the target y.
        """
        features = self.featurize_state(s)
        self.model.partial_fit([features], y)