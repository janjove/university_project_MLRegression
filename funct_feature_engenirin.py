def pre_or_post_2008_bool(row):
            if(row['YrSold'] < 2008) or (row['YrSold'] == 2008 and row['MoSold'] < 9):
                return '0'
            else:
                return '1'
def pre_or_post_2008(row):
        return row['YrSold'] - 2008 + (row['MoSold'] - 9)/12
def calificacio_a_numero(calificacion):
    map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
    return map.get(calificacion, 0)
def main_road(row):
    if (row['Condition1'] == 'Artery' or row['Condition2'] == 'Artery'):
        return '1'
    else:
        return '0'
def ad_station(row):
        if (row['Condition1'] == 'RRAn' or row['Condition2'] == 'RRAn' or row['Condition1'] == 'RRAe' or row['Condition2'] == 'RRAe'):
            return '1'
        else:
            return '0'
def air_condition_to_numeric(value):
        map = {'Y': 1, 'N': 0}
        return map.get(value, -1)
def functional_to_number(functionality):
    map = {
        'Typ': 0, 'Min1': 1, 'Min2': 2, 'Mod': 3,'Maj1': 4,'Maj2': 5,'Sev': 6,'Sal': 7
    }
    return map.get(functionality, -1)

def good_position(row):
    if (row['Condition1'] != 'Norm' or row['Condition2'] != 'Norm'):
        return '1'
    else:
        return '0'
    
def normal_sale(row):
    if (row['SaleCondition'] == 'Normal'):
        return '1'
    else:
        return '0'