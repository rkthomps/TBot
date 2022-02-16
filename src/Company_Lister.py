import pandas as pd

'''
Gets a list of companies for the scraper to analyze
'''
class Company_Lister:

    '''
    Return the df containing SNP 500 data
    '''
    def get_snp_df(self):
        snp_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        return pd.read_html(snp_url)[0]

    '''
    Returns a list of company symbols from the SNP 500
    '''
    def get_snp(self):
        return list(self.get_snp_df()['Symbol'])
    
    '''
    Returns a list of companies that have been in the SNP 500 
    for the given number of years
    '''
    def get_snp_since(self, date):
        snp_df = self.get_snp_df()
        date_name = 'Date first added'
        snp_df = snp_df.loc[~pd.isna(snp_df[date_name])]
        snp_df[date_name] = snp_df[date_name].apply(lambda x: x.split()[0])
        snp_df[date_name] = pd.to_datetime(snp_df[date_name])
        return list(snp_df.loc[snp_df[date_name] < date, 'Symbol'])









