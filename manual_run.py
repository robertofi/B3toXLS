from B3toXLS.main import updateNotasB3, parseFolder, path
parseFolder='/etc/tekton/B3toXLS/B3toXLS/app/toParse_1/'
path=''
dfOp, dfNotas = updateNotasB3(corretora='xp', path=path, parseFolder=parseFolder)
print(dfOp, dfNotas)



