import pandas as pd
import numpy as np
import seaborn as sns


gamma = 5
n = 100
period = ['60', '120', '180', '240', '300']

# Generate random data
np.random.seed(1234)
cryoRL_Res50 = np.r_[ np.random.normal(loc = 23.5, scale = 0 * gamma, size= n ) ,
                      np.random.normal(loc = 44.3, scale = 0 * gamma, size= n ) ,
                      np.random.normal(loc = 61.4, scale = 0 * gamma, size= n ) ,
                      np.random.normal(loc = 82.4, scale = 0 * gamma, size= n ),
                      np.random.normal(loc = 96.0, scale = 0 * gamma, size= n ) ]
cryoRL_Res50_sq = np.r_[ np.random.normal(loc = 23.3, scale = 0 * gamma, size= n ) ,
                      np.random.normal(loc = 44.4, scale = 0 * gamma, size= n ) ,
                      np.random.normal(loc = 61.9, scale = 0 * gamma, size= n ) ,
                      np.random.normal(loc = 83.2, scale = 0 * gamma, size= n ),
                      np.random.normal(loc = 94.6, scale = 0 * gamma, size= n ) ]
cryoRL_Res50_gr = np.r_[ np.random.normal(loc = 21.3, scale = 0 * gamma, size= n ) ,
                      np.random.normal(loc = 45.1, scale = 0 * gamma, size= n ) ,
                      np.random.normal(loc = 66.7, scale = 0 * gamma, size= n ) ,
                      np.random.normal(loc = 85.9, scale = 0 * gamma, size= n ),
                      np.random.normal(loc = 97.1, scale = 0 * gamma, size= n ) ]
cryoRL_Res50_sq_gr = np.r_[ np.random.normal(loc = 22.9, scale = 0 * gamma, size= n ) ,
                      np.random.normal(loc = 44.4, scale = 0 * gamma, size= n ) ,
                      np.random.normal(loc = 60.5, scale = 0 * gamma, size= n ) ,
                      np.random.normal(loc = 82.6, scale = 0 * gamma, size= n ),
                      np.random.normal(loc = 96.0, scale = 0 * gamma, size= n ) ]

# errors = [3.5,3.0,2.6,2.6, 
#           1.5,2.4,2.0,2.4,
#           0.9,0.6,1.7,2.3]

# Create a dataframe
df = pd.DataFrame({'Low CTF Counts': np.r_[cryoRL_Res50, cryoRL_Res50_sq, cryoRL_Res50_gr, cryoRL_Res50_sq_gr],
                   'Time': np.r_[np.repeat(period, n), np.repeat(period, n), np.repeat(period, n), np.repeat(period, n)],
                   'Method': np.repeat(['cryoRL_Res50 (0.23, 0.09)',  'cryoRL_Res50 (0.46, 0.09)', 'cryoRL_Res50 (0.23, 0.18)', 'cryoRL_Res50 (0.46, 0.18)'], len(period) * n),
                   'Subject': np.r_[np.tile(np.arange(n), 5),
                                    np.tile(np.arange(n, n + n), 5),
                                    np.tile(np.arange(2*n, 3*n), 5),
                                    np.tile(np.arange(3*n, 4*n), 5)]})

sns.set()
plot = sns.pointplot(data=df, x='Time', y='Low CTF Counts', hue='Method', dodge=True, #markers=['o', 's', 'x', 'p'],
	    capsize=.1, errwidth=1, palette='colorblind')
# plot.errorbar([])
plot.legend_.set_title(None)
fig = plot.get_figure()
fig.savefig("cryo_reward_effect.pdf")
