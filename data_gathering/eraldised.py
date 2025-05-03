from dbfread import DBF

# Load the DBF file
table = DBF('eraldis.dbf')

# Iterate over records
for record in table:
    print(record['ID'])
