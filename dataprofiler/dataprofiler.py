import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.stats import skew
import sys
import seaborn as sns
from scipy.cluster import hierarchy

def isMonocity(data):
    inc = all(data[i] <= data[i+1] for i in range(len(data)-1))
    dec = all(data[i] >= data[i+1] for i in range(len(data)-1))
    if inc:
        return 'Increasing'
    elif dec:
        return 'Decreasing'
    else:
        return 'None'

def profileReport(df: pd.DataFrame, title = 'Profiling Report'):
    raw_data = df
    title = title + '.html'

    # Create an HTML file and open it for writing
    with open(title, 'w') as f:
        # Write the HTML header
        f.write('<!DOCTYPE html>\n')
        f.write('<html>\n')
        f.write('<head>\n')
        f.write('<title>Data Profiling Tool</title>\n')
        f.write('</head>\n')
        f.write('<body>\n')
        f.write('<h1>Data Profiling Tool</h1>\n')
        # OVERVIEW
        f.write('<h2>Overview</h2>\n')
        f.write('<h3>Dataset statistic</h3>\n')
        f.write('<p><strong>Number of variables:</strong> {}</p>\n'.format(raw_data.shape[1]))
        f.write('<p><strong>Number of observations:</strong> {}</p>\n'.format(raw_data.shape[0]))
        f.write('<p><strong>Number of missing cells:</strong> {}</p>\n'.format(raw_data.isnull().sum().sum()))
        f.write('<p><strong>Missing cells(%):</strong> {}%</p>\n'.format(round(float(raw_data.isnull().sum().sum()) / float((raw_data.shape[0] * raw_data.shape[1])) * 100.0, 2)))
        f.write('<p><strong>Duplicated rows:</strong> {}</p>\n'.format(raw_data.duplicated().sum()))
        f.write('<p><strong>Duplicated rows(%):</strong> {}%</p>\n'.format(round(raw_data.duplicated().sum() / raw_data.shape[0], 2)))
        f.write('<p><strong>Total size in memory:</strong> {} KB</p>\n'.format(round(sys.getsizeof(raw_data) / 1024, 2)))
        f.write('<hr>\n')
    
        f.write('<h3>Variables type</h3>\n')
        f.write('<p><strong>CAT:</strong> {}</p>\n'.format((raw_data.dtypes == 'object').sum()))
        f.write('<p><strong>NUM:</strong> {}</p>\n'.format((raw_data.dtypes != 'object').sum()))
        f.write('<hr>\n')
        
        f.write('<h2>Variables</h2>\n')
    
        # Iterate over each column
        for col in raw_data.columns:
            # VARIABLES
            f.write('<h3 style="color: blue;">{}</h3>\n'.format(col))
            if raw_data[col].dtype == 'object':
                f.write('<p style="color: grey;"><i>Categorial</i></p>\n')
            else:
                f.write('<p style="color: grey;"><i>Numerical</i></p>\n')
            f.write('<h3><i>Overview</i></h3>\n')
            f.write('<p><strong>Distinct:</strong> {}</p>\n'.format(len(raw_data[col].unique())))
            f.write('<p><strong>Distinct(%):</strong> {}%</p>\n'.format(round(len(raw_data[col].unique()) / raw_data[col].shape[0] * 100.0, 2)))
            f.write('<p><strong>Missing:</strong> {}</p>\n'.format(raw_data[col].isnull().sum()))
            f.write('<p><strong>Missing(%):</strong> {}%</p>\n'.format(round(raw_data[col].isnull().sum() / raw_data[col].shape[0] * 100.0, 2)))
    
            if raw_data[col].dtype != 'object':
                f.write('<p><strong>Type:</strong> Numerical</p>\n')
                f.write('<p><strong>Infinity:</strong> {}</p>\n'.format(np.isinf(raw_data[col]).sum()))
                f.write('<p><strong>Infinity(%):</strong> {}%</p>\n'.format(round(np.isinf(raw_data[col]).sum() / raw_data[col].shape[0] * 100.0, 2)))
                f.write('<p><strong>Mean:</strong> {}</p>\n'.format(raw_data[col].mean()))
                f.write('<p><strong>Min:</strong> {}</p>\n'.format(raw_data[col].min()))
                f.write('<p><strong>Max:</strong> {}</p>\n'.format(raw_data[col].max()))
                f.write('<p><strong>Zeros:</strong> {}</p>\n'.format((raw_data[col] == 0).sum()))
                f.write('<p><strong>Zeros(%):</strong> {}%</p>\n'.format(round(((raw_data[col] == 0).sum()) / raw_data[col].shape[0] * 100.0, 2)))
            
            
            f.write('<p><strong>Memory size:</strong> {} KB</p>\n'.format(round(sys.getsizeof(raw_data[col]) / 1024, 2)))
            f.write('<hr>\n')
    
            # VARIABLES
            if raw_data[col].dtype != 'object':
                # Quantile statistics
                f.write('<h3><i>Quantile statistics</i></h3>\n')
                f.write('<p><strong>Min:</strong> {}</p>\n'.format(raw_data[col].min()))
                f.write('<p><strong>5th percentile:</strong> {}</p>\n'.format(raw_data[col].quantile(0.05)))
                f.write('<p><strong>25th percentile:</strong> {}</p>\n'.format(raw_data[col].quantile(0.25)))
                f.write('<p><strong>Median:</strong> {}</p>\n'.format(raw_data[col].median()))
                f.write('<p><strong>75th percentile:</strong> {}</p>\n'.format(raw_data[col].quantile(0.75)))
                f.write('<p><strong>95th percentile:</strong> {}</p>\n'.format(raw_data[col].quantile(0.95)))
                f.write('<p><strong>Max:</strong> {}</p>\n'.format(raw_data[col].max()))
                f.write('<p><strong>Range:</strong> {}</p>\n'.format(raw_data[col].max() - raw_data[col].min()))
                f.write('<p><strong>IQR:</strong> {}</p>\n'.format(raw_data[col].quantile(0.75) - raw_data[col].quantile(0.25)))
                f.write('<hr>\n')
    
                # Descriptive statistics
                f.write('<h3><i>Descriptive statistics</i></h3>\n')
                f.write('<p><strong>Standard deviation:</strong> {}</p>\n'.format(raw_data[col].std()))
                f.write('<p><strong>Coefficient of variation (CV):</strong> {}%</p>\n'.format(raw_data[col].std() / raw_data[col].mean() * 100))
                f.write('<p><strong>Kurtosis:</strong> {}</p>\n'.format(kurtosis(raw_data[col])))
                f.write('<p><strong>Mean:</strong> {}</p>\n'.format(raw_data[col].mean()))
                f.write('<p><strong>Mean absolute deviation (MAD):</strong> {}</p>\n'.format(np.median(np.abs(raw_data[col] - raw_data[col].median()))))
                f.write('<p><strong>Skewness:</strong> {}</p>\n'.format(skew(raw_data[col])))
                f.write('<p><strong>Sum:</strong> {}</p>\n'.format(raw_data[col].sum()))
                f.write('<p><strong>Variance:</strong> {}</p>\n'.format(raw_data[col].var()))
                f.write('<p><strong>Monocity:</strong> {}</p>\n'.format(isMonocity(raw_data[col])))
                f.write('<hr>\n')
    
                # Histogram
                f.write('<h3><i>Histogram</i></h3>\n')
                plt.figure()
                plt.hist(raw_data[col], bins = 50)
                plt.ylabel('Frequency')
                plt.savefig(f'hist_{col}.png')
                plt.close()
                f.write(f'<img src="hist_{col}.png" alt="Histogram">\n')
                f.write('<hr>\n')
    
                # Common values
                f.write('<h3><i>Common values</h3></i>\n')
                cnt = raw_data[col].value_counts()
                df_cnt = pd.DataFrame({
                    'Value': cnt.index,
                    'Count': cnt.values,
                    'Frequency': (cnt.values/len(raw_data[col])*100).round(2)
                })
                df_html = df_cnt.head(10).to_html(index = False)
                f.write(df_html)
                f.write('<hr>\n')
    
                # Extreme values
                f.write('<h3><i>Extreme values</h3></i>\n')
                f.write('<p><strong>Minimum 5 values</strong></p>\n')
                df_html = df_cnt.nsmallest(5, 'Value').to_html(index = False)
                f.write(df_html)
                f.write('<p><strong>Maximum 5 values</strong></p>\n')
                df_html = df_cnt.nlargest(5, 'Value').to_html(index = False)
                f.write(df_html)
                f.write('<hr>\n')
            else:
                # Frequencies
                f.write('<h3><i>Frequencies</h3></i>\n')
                f.write('<p><strong>Common values</strong></p>\n')
                cnt = raw_data[col].value_counts()
                df_cnt = pd.DataFrame({
                    'Value': cnt.index,
                    'Count': cnt.values,
                    'Frequency': (cnt.values/len(raw_data[col])*100).round(2)
                })
                df_html = df_cnt.head(10).to_html(index = False)
                f.write(df_html)
                f.write('<hr>\n')
                
                f.write('<p><strong>Overview</strong></p>\n')
                f.write('<p><strong>Unique:</strong> {}</p>\n'.format(len(raw_data[col].unique())))
                f.write('<p><strong>Unique(%):</strong> {}%</p>\n'.format(round(len(raw_data[col].unique())/len(raw_data[col])*100.0, 2)))
                
                freq_cnt = df_cnt['Count'].value_counts()
    
                plt.figure()
                plt.bar(freq_cnt.index, freq_cnt.values)
                plt.savefig(f'hist_{col}.png')
                plt.close()
                f.write(f'<img src="hist_{col}.png" alt="Histogram">\n')
                f.write('<hr>\n')
    
                if len(raw_data[col].unique()) <= 7:
                    f.write('<p><strong>Chart</strong></p>\n')
                    plt.figure()
                    myLabels = raw_data[col].value_counts().index
                    plt.pie(raw_data[col].value_counts().values, labels = myLabels, autopct = '%1.1f%%')
                    plt.savefig(f'pie_{col}.png')
                    plt.close()
                    f.write(f'<img src="pie_{col}.png" alt="Pie chart">\n')
                    f.write('<hr>\n')
    
                # Length            
                f.write('<h3><i>Length</i></h3>\n')
                lengths = raw_data[col].str.len()
                grouped_by_length = lengths.value_counts()
                
                f.write('<p><strong>Max length:</strong> {}</p>\n'.format(lengths.max()))
                f.write('<p><strong>Median length:</strong> {}</p>\n'.format(lengths.median()))
                f.write('<p><strong>Mean length:</strong> {:.2f}</p>\n'.format(lengths.mean()))
                f.write('<p><strong>Min length:</strong> {}</p>\n'.format(lengths.min()))
                plt.figure()
                plt.bar(grouped_by_length.index, grouped_by_length.values)
                plt.savefig(f'length_{col}.png')
                plt.close()
                f.write(f'<img src="length_{col}.png" alt="Length histogram chart">\n')
                
                f.write('<hr>\n')
        
        # INTERACTIONS
        f.write('<h2>Interactions</h2>\n')
        cols = raw_data.select_dtypes(exclude = ['object']).columns
        for i in cols:
            for j in cols:
                f.write(f'<h3><i>{i} - {j}</i></h3>')
                plt.figure()
                plt.scatter(raw_data[i], raw_data[j])
                plt.savefig(f'scatter_{i}_{j}.png')
                plt.close()
                f.write(f'<img src="scatter_{i}_{j}.png" alt="Scatter plot">\n')
                f.write('<hr>\n')
    
        # CORRELATIONS
        f.write('<h2>Correlations</h2>\n')
        methods = ['pearson', 'spearman', 'kendall']
        numerical_df = raw_data.select_dtypes(exclude = ['object'])
        
        for med in methods:
            if med == 'pearson':
                f.write("<h3><i>Pearson's r</i></h3>\n")
            elif med == 'spearman':
                f.write("<h3><i>Spearman's p</i></h3>\n")
            else:
                f.write("<h3><i>Kendall's t</i></h3>\n")
            
            plt.figure()
            corr = numerical_df.corr(method = med)
            sns.heatmap(corr)
            plt.savefig(f'corr_{med}.png')
            plt.close()
    
            f.write('<table><tr>')
            f.write(f'<td><img src="corr_{med}.png" alt="Correlations"></td>')
            if med == 'pearson':
                f.write("<td><h2 style='color:grey;'><strong>Pearson's r</strong></h4><p style='color:grey;'><i>The Pearson's correlation coefficient (r) is a measure of linear correlation between two variables. It's value lies between -1 and +1, -1 indicating total negative linear correlation, 0 indicating no linear correlation and 1 indicating total positive linear correlation. Furthermore, r is invariant under separate changes in location and scale of the two variables, implying that for a linear function the angle to the x-axis does not affect r.</i></p><p></p><p style='color:grey;'><i>To calculate r for two variables X and Y, one divides the covariance of X and Y by the product of their standard deviations.</i></p></td>")
            elif med == 'spearman':
                f.write("<td><h2 style='color:grey;'><strong>Spearman's p</strong></h4><p style='color:grey;'><i>The Spearman's rank correlation coefficient (p) is a measure of monotonic correlation between two variables, and is therefore better in catching nonlinear monotonic correlations than Pearson's r. It's value lies between -1 and +1, -1 indicating total negative monotonic correlation, 0 indicating no monotonic correlation and 1 indicating total positive monotonic correlation.</i></p><p></p><p style='color:grey;'><i>To calculate p for two variables X and Y, one divides the covariance of the rank variables of X and Y by the product of their standard deviations.</i></p></td>")
            else:
                f.write("<td><h2 style='color:grey;'><strong>Kendall's t</strong></h4><p style='color:grey;'><i>Similarly to Spearman's rank correlation coefficient, the Kendall rank correlation coefficient (t) measures ordinal association between two variables. It's value lies between -1 and +1, -1 indicating total negative correlation, 0 indicating no correlation and 1 indicating total positive correlation.</i></p><p></p><p style='color:grey;'><i>To calculate t for two variables X and Y, one determines the number of concordant and discordant pairs of observations. t is given by the number of concordant pairs minus the discordant pairs divided by the total number of pairs.</i></p></td>")
            f.write('</tr></table>\n')
            f.write('<hr>\n')
    
        # MISSING VALUES
        f.write('<h2>Missing values</h2>\n')
        f.write('<h3><i>Count</i></h3>\n')
    
        # Bar chart
        missing_data = raw_data.isnull().sum()
        plt.bar(missing_data.index, missing_data.values)
        plt.xticks(rotation = 45, ha = 'right')
        plt.tight_layout()
        plt.savefig('missing_bar.png')
        plt.close()
        f.write('<img src="missing_bar.png" alt="matrix chart">')
        f.write('<hr>\n')
    
        # Matrix chart
        f.write('<h3><i>Matrix</i></h3>\n')
        sns.heatmap(raw_data.isnull(), cmap = 'viridis')
        plt.savefig('missing_matrix.png')
        plt.close()
        f.write('<img src="missing_matrix.png" alt="matrix chart">')
        f.write('<hr>\n')
    
        # Dendrogram
        f.write('<h3><i>Dendrogram</i></h3>\n')
        distance_matrix = hierarchy.distance.pdist(raw_data.isnull().T, metric = 'euclidean')
        Z = hierarchy.linkage(distance_matrix, method = 'complete')
        
        dendrogram = hierarchy.dendrogram(Z, labels = raw_data.columns)
        plt.xticks(rotation = 45, ha = 'right')
        plt.savefig('missing_dendrogram.png')
        plt.close()
        f.write('<img src="missing_dendrogram.png" alt="matrix chart">')
        f.write('<hr>\n')
    
        # SAMPLE
        f.write('<h2>Sample</h2>\n')
        f.write('<h3><i>First 10 rows</i></h3>\n')
        df_html = raw_data.head(10).to_html(index = False)
        f.write(df_html)
    
        f.write('<h3><i>Last 10 rows</i></h3>\n')
        df_html = raw_data.tail(10).to_html(index = False)
        f.write(df_html)    
        
        # Write the HTML footer
        f.write('</body>\n')
        f.write('</html>\n')
