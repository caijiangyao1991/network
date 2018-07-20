from neo4j.v1 import GraphDatabase, basic_auth
import pandas as pd
#向数据库对象请求一个新的驱动程序
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri,auth=basic_auth("neo4j","q6594461"))
#向驱动程序对象请求一个新会话
session = driver.session()
#请求会话对象创建事务
result = session.run("match (accountHolder:AccountHolder)-[]->(contactInformation)  with contactInformation,count(accountHolder) as RingSize match (contactInformation)<-[]-(accountHolder), (accountHolder)-[r:HAS_CREDITCARD|HAS_UNSECUREDLOAN]->(unsecuredAccount) with collect(accountHolder.UinqueId) as AccountHolders, contactInformation, RingSize, sum(case type(r) when 'HAS_CREDITCARD' THEN unsecuredAccount.Limit when 'HAS_UNSECUREDLOAN' THEN unsecuredAccount.Balance ELSE 0 END) as FinancialRisk where RingSize>1 return AccountHolders, labels(contactInformation) as ContactType, RingSize, round(FinancialRisk) as FinancialRisk order by RingSize DESC")
data = {"AccountHolders":[],"ContactType":[],"RingSize":[],"FinancialRisk":[]}
for record in result:
    # print("%s %s" (record["AccountHolders"], record["ContactType"], record["RingSize"]))
    data["AccountHolders"].append(record["AccountHolders"])
    data["ContactType"].append(record["ContactType"])
    data["RingSize"].append(record["RingSize"])
    data["FinancialRisk"].append(record["FinancialRisk"])
data1 = pd.DataFrame(data)
print(data1)