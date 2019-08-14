class ItemKNN:
    '''
    Item-to-item predictor that computes the the similarity to all items to the given item.
    '''    
    
    def __init__(self, n_sims = 100, lmbd = 20, alpha = 0.5, session_key = 'SessionId', item_key = 'ItemId', time_key = 'Time'):
        self.n_sims = n_sims
        self.lmbd = lmbd
        self.alpha = alpha

    def fit(self, data):
        '''
        Trains the predictor.
        '''
        data.set_index(np.arange(len(data)), inplace=True)
        itemids = data['product_id'].unique()
        n_items = len(itemids) 
        data = pd.merge(data, pd.DataFrame({'product_id':itemids, 'ItemIdx':np.arange(len(itemids))}), on='product_id', how='inner')
        sessionids = data['order_id'].unique()
        data = pd.merge(data, pd.DataFrame({'order_id':sessionids, 'SessionIdx':np.arange(len(sessionids))}), on='order_id', how='inner')
        supp = data.groupby('SessionIdx').size()
        session_offsets = np.zeros(len(supp)+1, dtype=np.int32)
        session_offsets[1:] = supp.cumsum()
        index_by_sessions = data.sort_values(['SessionIdx', 'add_to_cart_order']).index.values
        supp = data.groupby('ItemIdx').size()
        item_offsets = np.zeros(n_items+1, dtype=np.int32)
        item_offsets[1:] = supp.cumsum()
        index_by_items = data.sort_values(['ItemIdx', 'add_to_cart_order']).index.values
        self.sims = dict()
        for i in range(n_items):
            iarray = np.zeros(n_items)
            start = item_offsets[i]
            end = item_offsets[i+1]
            for e in index_by_items[start:end]:
                uidx = data.SessionIdx.values[e]
                ustart = session_offsets[uidx]
                uend = session_offsets[uidx+1]
                user_events = index_by_sessions[ustart:uend]
                iarray[data.ItemIdx.values[user_events]] += 1
            iarray[i] = 0
            norm = np.power((supp[i] + self.lmbd), self.alpha) * np.power((supp.values + self.lmbd), (1.0 - self.alpha))
            norm[norm == 0] = 1
            iarray = iarray / norm
            indices = np.argsort(iarray)[-1:-1-self.n_sims:-1]
            self.sims[itemids[i]] = pd.Series(data=iarray[indices], index=itemids[indices])
    
    def predict_next(self, session_id, input_item_id, predict_for_item_ids):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
        '''
        preds = np.zeros(len(predict_for_item_ids))
        sim_list = self.sims[input_item_id]
        mask = np.in1d(predict_for_item_ids, sim_list.index)
        preds[mask] = sim_list[predict_for_item_ids[mask]]
        return pd.Series(data=preds, index=predict_for_item_ids)