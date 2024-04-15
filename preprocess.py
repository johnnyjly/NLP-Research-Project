import pandas as pd
import re

'''
Some Data preprocessing and Data Visual Analysis

Columns of dataset: 
    
    String columns: 
        'Job Title', 'Job Description', 'Location', 
        'Headquarters', 'company_txt(company name)',
        'Founded(originally number but consider it str)'
    
    Category Columns: 
        'Size', 'Type of ownership', 'Industry', 'Sector',
        'Degree(M for masters, P for PhD)'        

    Bool Columns(1=Required skill/certificate): 
        'Python', 'spark', 'aws', 
        'excel', 'sql', 'sas', 'keras', 'pytorch', 'scikit', 'tensor', 'hadoop', 
        'tableau', 'bi', 'flink', 'mongo', 'google_an',

    Columns to Drop: 
        'Rating', 'Salary Estimate(duplicate)', 'Revenue', 'Competitors', 
        'Hourly', 'Employer provided', 
        'Company Name(because this includes the user rating)', 
        'Job Location(duplicate)', 'job_title_sim(duplicate)', 
        'seniority_by_title(not actually useful)', 
        'index', 'Age(of company, useless when we have founded year)',
        'Avg Salary(K) (In application, we won't be inputing this)', 

    Target Columns: 
         'Lower Salary(k)', 'Upper Salary(k)'
         # One idea is to predict lower first, 
         # then add lower to input and predict upper

'''

def preprocess_data(data_path):

    # Step 0: Load data
    data = pd.read_csv(data_path)
    data = data.dropna().drop(columns=['Rating', 'Revenue', 
            'Competitors', 'Hourly', 'Employer provided', 'Company Name', 
            'Job Location', 'job_title_sim', 'seniority_by_title',
            'Salary Estimate', 'Age', 'Avg Salary(K)'])
    
    data.set_index('index', inplace=True)

    # Step 1: Concatenate the string parts
    data['string'] = data['Job Title'] + '-' +\
        data['Job Description'] + '-' +\
        data['Location'] + '-' +\
        data['Headquarters'] + '-' +\
        data['company_txt'] + '-' +\
        data['Founded'].astype(str)
    
    # Step 2: Combine Bool columns into one number, converted from binary
    skill_list = ['Python', 'spark', 'aws', 'excel', 'sql', 'sas', 'keras',
                   'pytorch', 'scikit', 'tensor', 'hadoop', 'tableau', 'bi',
                   'flink', 'mongo', 'google_an']
    # Note: The order of skills is inverted to digits of binary number
    #       e.g. 0b0000000000000001 implies only requiring Python
    #            0b1000000000000000 implies only requiring google_an
    data['skill_str'] = 'Required Skills:'
    for skill in skill_list:
        data.loc[data[skill], 'skill_str'] += '{};'.format(skill)
    data.loc[data['skill_str']=='Required Skills:', 'skill_str'] = ''
    
    # Step 3: Convert categorical columns into numerical values
    data.rename(columns={'Size': 'Company Size'}, inplace=True)
    category_list = ['Company Size', 'Type of ownership', 'Industry', 
                          'Sector', 'Degree']
    data['category_str'] = ''
    for category in category_list:
        data['category_str'] = data.apply(lambda x: '{}{}: {};'.format(x.category_str, category, x[category]), axis=1)
    data['string'] = data['string'] + data['skill_str'] + data['category_str']

    data['string'] = data.apply(lambda x: pre_process_text(x['string']), axis=1)
    
    # Step 4: Create target columns
    data.rename(columns={'Lower Salary': 'target_l', 'Upper Salary': 'target_u'}, inplace=True)

    # Step 5: Select Processed Outputs
    output_cols = ['string', 'target_l', 'target_u']
    # output_cols = ['string', 'target_l']
    data[output_cols].to_csv('./data/preprocessed.csv', index=False)
    return data[output_cols]


def pre_process_text(text):
  text = re.sub("[^a-zA-Z]", " ", text)
  text = text.lower()
  tokens = text.split()
  return " ".join(tokens)


if __name__ == '__main__':
    preprocess_data('./data/data_cleaned_2021.csv')