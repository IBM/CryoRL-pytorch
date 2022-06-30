import pandas as pd
import numpy as np
import seaborn as sns


gamma = 1.0
n = 100
#period = ['60', '120', '180', '240', '300', '360']
period = ['120', '240', '360', '480', '600']

# Generate random data
np.random.seed(1234)
'''
cryoRL_GT = np.r_[ np.random.normal(loc = 24.7, scale = 3.5 * gamma, size= n ) ,
                   np.random.normal(loc = 48.9, scale = 3.0 * gamma, size= n ) ,
                   np.random.normal(loc = 73.2, scale = 2.6 * gamma, size= n ) ,
                   np.random.normal(loc = 95.7, scale = 2.6 * gamma, size= n ),
                   np.random.normal(loc = 118.8, scale = 3.4 * gamma, size= n ) ]
greedy_GT = np.r_[ np.random.normal(loc = 23.4, scale = 2.1 * gamma, size= n ) ,
                   np.random.normal(loc = 51.2, scale = 1.8 * gamma, size= n ) ,
                   np.random.normal(loc = 73.9, scale = 4.1 * gamma, size= n ) ,
                   np.random.normal(loc = 96.5, scale = 2.3 * gamma, size= n ),
                   np.random.normal(loc = 119.4, scale = 3.8 * gamma, size= n ) ]
cryoRL_Res18 = np.r_[ np.random.normal(loc = 22.9, scale = 1.5 * gamma, size= n ) ,
                      np.random.normal(loc = 42.9, scale = 2.4 * gamma, size= n ) ,
                      np.random.normal(loc = 58.0, scale = 2.0 * gamma, size= n ) ,
                      np.random.normal(loc = 80.0, scale = 2.4 * gamma, size= n ) ,
                      np.random.normal(loc = 92.3, scale = 1.7 * gamma, size= n ) ]
cryoRL_Res50 = np.r_[ np.random.normal(loc = 23.5, scale = 0.9 * gamma, size= n ) ,
                      np.random.normal(loc = 44.3, scale = 0.6 * gamma, size= n ) ,
                      np.random.normal(loc = 61.4, scale = 1.7 * gamma, size= n ) ,
                      np.random.normal(loc = 82.4, scale = 2.3 * gamma, size= n ),
                      np.random.normal(loc = 96.0, scale = 3.2 * gamma, size= n ) ]
'''

'''
greedy_GT = np.r_[ np.random.normal(loc = 23.4, scale = 2.1 * gamma, size= n ) ,
                   np.random.normal(loc = 51.2, scale = 1.8 * gamma, size= n ) ,
                   np.random.normal(loc = 72.9, scale = 4.1 * gamma, size= n ) ,
                   np.random.normal(loc = 96.5, scale = 2.3 * gamma, size= n ) ,
                   np.random.normal(loc = 119.4, scale = 3.8 * gamma, size= n ),
                   np.random.normal(loc = 139.1, scale = 2.3 * gamma, size= n ) ]

greedy_Res18 = np.r_[ np.random.normal(loc = 18.2, scale = 2.8 * gamma, size= n ) ,
                   np.random.normal(loc = 40.8, scale = 3.1 * gamma, size= n ) ,
                   np.random.normal(loc = 59.3, scale = 6.5 * gamma, size= n ) ,
                   np.random.normal(loc = 68.4, scale = 4.6 * gamma, size= n ) ,
                   np.random.normal(loc = 86.2, scale = 3.9 * gamma, size= n ),
                   np.random.normal(loc = 103.1, scale = 2.7 * gamma, size= n ) ]
greedy_Res50 = np.r_[ np.random.normal(loc = 21.3, scale = 2.1 * gamma, size= n ) ,
                   np.random.normal(loc = 43.4, scale = 2.1 * gamma, size= n ) ,
                   np.random.normal(loc = 56.8, scale = 5.2 * gamma, size= n ) ,
                   np.random.normal(loc = 72.1, scale = 3.2 * gamma, size= n ) ,
                   np.random.normal(loc = 91.9, scale = 2.9 * gamma, size= n ),
                   np.random.normal(loc = 107.8, scale = 4.9 * gamma, size= n ) ]

cryoRL_GT = np.r_[ np.random.normal(loc = 24.6, scale = 3.1 * gamma, size= n ) ,
                   np.random.normal(loc = 48.3, scale = 3.3 * gamma, size= n ) ,
                   np.random.normal(loc = 72.8, scale = 3.1 * gamma, size= n ) ,
                   np.random.normal(loc = 96.8, scale = 2.4 * gamma, size= n ),
                   np.random.normal(loc = 119.3, scale = 3.5 * gamma, size= n ),
                   np.random.normal(loc = 142.2, scale = 3.9 * gamma, size= n ) ]
cryoRL_Res18 = np.r_[ np.random.normal(loc = 22.9, scale = 2.6 * gamma, size= n ) ,
                      np.random.normal(loc = 45.4, scale = 2.8 * gamma, size= n ) ,
                      np.random.normal(loc = 68.5, scale = 3.2 * gamma, size= n ) ,
                      np.random.normal(loc = 90.7, scale = 3.1 * gamma, size= n ) ,
                      np.random.normal(loc = 110.5, scale = 2.5 * gamma, size= n ) ,
                      np.random.normal(loc = 129.7, scale = 4.4 * gamma, size= n ) ]
cryoRL_Res50 = np.r_[ np.random.normal(loc = 22.6, scale = 2.7 * gamma, size= n ) ,
                      np.random.normal(loc = 47.5, scale = 2.9 * gamma, size= n ) ,
                      np.random.normal(loc = 70.8, scale = 4.2 * gamma, size= n ) ,
                      np.random.normal(loc = 92.9, scale = 3.6 * gamma, size= n ),
                      np.random.normal(loc = 111.0, scale = 5.1 * gamma, size= n ),
                      np.random.normal(loc = 133.2, scale = 2.6 * gamma, size= n ) ]
'''

greedy_GT = np.r_[ np.random.normal(loc = 49.4, scale = 2.8 * gamma, size= n ) ,
                   np.random.normal(loc = 92.9, scale = 3.1 * gamma, size= n ) ,
                   np.random.normal(loc = 133.7, scale = 3.1 * gamma, size= n ),
                   np.random.normal(loc = 181.5, scale = 4.3 * gamma, size= n ),
                   np.random.normal(loc = 226.8, scale = 3.7 * gamma, size= n ) ]


greedy_Res18 = np.r_[ np.random.normal(loc = 39.0, scale = 3.6 * gamma, size= n ) ,
                   np.random.normal(loc = 66.4, scale = 5.6 * gamma, size= n ) ,
                   np.random.normal(loc = 100.0, scale = 4.2 * gamma, size= n ) ,
                   np.random.normal(loc = 134.6, scale = 3.6 * gamma, size= n ),
                   np.random.normal(loc = 180.3, scale = 3.7 * gamma, size= n ) ]

greedy_Res50 = np.r_[ np.random.normal(loc = 41.8, scale = 2.5 * gamma, size= n ) ,
                   np.random.normal(loc = 69.3, scale = 3.2 * gamma, size= n ) ,
                   np.random.normal(loc = 104.9, scale = 4.9 * gamma, size= n ) ,
                   np.random.normal(loc = 147.9, scale = 5.1 * gamma, size= n ),
                   np.random.normal(loc = 190.0, scale = 5.4 * gamma, size= n ) ]

cryoRL_GT = np.r_[ np.random.normal(loc = 44.7, scale = 4.5 * gamma, size= n ) ,
                   np.random.normal(loc = 93.7, scale = 4.6 * gamma, size= n ) ,
                   np.random.normal(loc = 131.2, scale = 7.3 * gamma, size= n ) ,
                   np.random.normal(loc = 148.0, scale = 2.4 * gamma, size= n ),
                   np.random.normal(loc = 159.2, scale = 9.1 * gamma, size= n ) ]


cryoRL_Res18 = np.r_[ np.random.normal(loc = 45.1, scale = 3.8 * gamma, size= n ) ,
                      np.random.normal(loc = 84.3, scale = 3.2 * gamma, size= n ) ,
                      np.random.normal(loc = 114.8, scale = 4.7 * gamma, size= n ) ,
                      np.random.normal(loc = 154.3, scale = 4.4 * gamma, size= n ) ,
                      np.random.normal(loc = 193.4, scale = 4.1 * gamma, size= n ) ]


cryoRL_Res50 = np.r_[ np.random.normal(loc = 41.1, scale = 2.5 * gamma, size= n ) ,
                      np.random.normal(loc = 87.5, scale = 2.0 * gamma, size= n ) ,
                      np.random.normal(loc = 130.0, scale = 4.2 * gamma, size= n ) ,
                      np.random.normal(loc = 165.5, scale = 3.3 * gamma, size= n ),
                      np.random.normal(loc = 198.6, scale = 4.8 * gamma, size= n ) ]


# errors = [3.5,3.0,2.6,2.6,
#           1.5,2.4,2.0,2.4,
#           0.9,0.6,1.7,2.3]

# Create a dataframe
df = pd.DataFrame({'#lCTFs': np.r_[greedy_Res18, greedy_Res50, cryoRL_Res18, cryoRL_Res50],
                   'Time': np.r_[np.repeat(period, n), np.repeat(period, n), np.repeat(period, n), np.repeat(period, n)],
                   'Method': np.repeat(['greedy_R18', 'greedy_R50', 'cryoRL_R18', 'cryoRL_R50'], len(period) * n),
                   'Subject': np.r_[np.tile(np.arange(n), 5),
                                    np.tile(np.arange(n, n + n), 5),
                                    np.tile(np.arange(2*n, 3*n), 5),
                                    #np.tile(np.arange(3*n, 4*n), 5),
                                    #np.tile(np.arange(4*n, 5*n), 5),
                                    np.tile(np.arange(3*n, 4*n), 5)]})

sns.set()
plot = sns.pointplot(data=df, x='Time', y='#lCTFs', hue='Method', dodge=True,
        #markers=['o', 's', 'x', 'p', 'v', '*'],
        #linestyles=['--', '--', '--', '-', '-', '-'],
        markers=['o', 's', 'x', 'p'],
        linestyles=['--', '--',  '-', '-'],
	    capsize=.1, errwidth=1, palette='colorblind', ci="sd")

# plot.errorbar([])
plot.legend_.set_title(None)
fig = plot.get_figure()
fig.savefig("main-results.png")
fig.savefig("main-results.svg")
