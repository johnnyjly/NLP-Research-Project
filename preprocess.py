import pandas as pd
import torch
import os
import bert

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
            'Job Location', 'job_title_sim', 'seniority_by_title', 'index', 
            'Salary Estimate', 'Age', 'Avg Salary(K)'])

    # Step 1: Concatenate the string parts
    data['string'] = data['Job Title'] + '-' +\
        data['Job Description'] + '-' +\
        data['Location'] + '-' +\
        data['Headquarters'] + '-' +\
        data['company_txt'] + '-' +\
        str(data['Founded'])
    
    # Step 2: Combine Bool columns into one number, converted from binary
    skill_list = ['Python', 'spark', 'aws', 'excel', 'sql', 'sas', 'keras',
                   'pytorch', 'scikit', 'tensor', 'hadoop', 'tableau', 'bi',
                   'flink', 'mongo', 'google_an']
    # Note: The order of skills is inverted to digits of binary number
    #       e.g. 0b0000000000000001 implies only requiring Python
    #            0b1000000000000000 implies only requiring google_an
    data['skills'] = 0
    for skill in skill_list:
        data['skills'] += data[skill] * (2 ** (skill_list.index(skill)))
    
    # Step 3: Convert categorical columns into numerical values
    category_list = ['Size', 'Type of ownership', 'Industry', 
                          'Sector', 'Degree']
    category_ref = {}
    for category in category_list:
        data[category] = data[category].astype('category')
        category_ref[category] = data[category].cat.categories
        data[category] = data[category].cat.codes

    # Step 4: Create target columns
    data['target_l'] = data['Lower Salary'] 
    data['target_u'] = data['Upper Salary']

    # Step 5: Select Processed Outputs
    output_cols = ['string', 'skills', 'Size', 'Type of ownership', 'Industry', 
                   'Sector', 'Degree', 'target_l', 'target_u']
    
    return (skill_list, category_list, data[output_cols])

preprocess_data('./data/data_cleaned_2021.csv')