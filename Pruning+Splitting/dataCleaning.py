from eventmakerTryServer import eventMaker
import json
# change StanfordCoreNLP model location in eventmakerTryServer.py file if necessary

class parseLine(object):

	def __init__(self):
		self.splittedCC = []
		self.takeSBAR = []
		self.removedPP = []
		self.removedPunc = []
		self.parsedSent = []

	def getTree(self,inp):
		maker = eventMaker(inp)
		out = maker.callStanford(inp)
		out = json.loads(out)
		parseTreeStr = out["sentences"][0]["parse"]
		res = parseTreeStr + ";" + inp
		self.parsedSent.append(res)

	def splitCC(self):#,line):
		# This is a function that can split CC conjunction between Ss.
		for line in self.parsedSent:
			if "<EOS>" in line:
				self.splittedCC.append(line)

			else:
				sec = line.split(");")
				str1 = sec[0]+")"
				str2 = sec[1]

				if "(CC and) (S " in str1:
					sent = []
					split = str1.split("(CC and) (S ")

					for i, clause in enumerate(split):
						subsent = []
						#print(clause)
						for element in split[i].split(" "):
							if ")" in element:
								subsent.append(element.replace(")",""))
						if i != len(split)-1 and i != 0:
							first = " ".join(subsent)  # dealing with the period "."
							if first[-1].isalpha() == False:
								first = first[:-1]+"."
							else:
								first = first+"."
							sent.append("(S "+clause+"@@@@"+first)
							#print([clause,first])
						elif i != len(split)-1 and i == 0:
							first = " ".join(subsent)  # dealing with the period "."
							if first[-1].isalpha() == False:
								first = first[:-1]+"."
							else:
								first = first+"."
							sent.append(clause+"@@@@"+first)
						else:
							sent.append("(S "+clause+"@@@@"+" ".join(subsent))

					self.splittedCC = sent


				elif "(CC but) (S " in str1:
					sent = []
					split = str1.split("(CC but) (S ")

					for i, clause in enumerate(split):
						subsent = []
						for element in split[i].split(" "):
							if ")" in element:
								subsent.append(element.replace(")",""))
						if i != len(split)-1 and i != 0:
							first = " ".join(subsent)  # dealing with the period "."
							if first[-1].isalpha() == False:
								first = first[:-1]+"."
							else:
								first = first+"."
							sent.append("(S "+clause+"@@@@"+first)
							#print([clause,first])
						elif i != len(split)-1 and i == 0:
							first = " ".join(subsent)  # dealing with the period "."
							if first[-1].isalpha() == False:
								first = first[:-1]+"."
							else:
								first = first+"."
							sent.append(clause+"@@@@"+first)
						else:
							sent.append("(S "+clause+"@@@@"+" ".join(subsent))

					self.splittedCC = sent


				elif "(CC or) (S " in str1:
					sent = []
					split = str1.split("(CC or) (S ")

					for i, clause in enumerate(split):
						subsent = []
						for element in split[i].split(" "):
							if ")" in element:
								subsent.append(element.replace(")",""))
						if i != len(split)-1 and i != 0:
							first = " ".join(subsent)  # dealing with the period "."
							if first[-1].isalpha() == False:
								first = first[:-1]+"."
							else:
								first = first+"."
							sent.append("(S "+clause+"@@@@"+first)
							#print([clause,first])
						elif i != len(split)-1 and i == 0:
							first = " ".join(subsent)  # dealing with the period "."
							if first[-1].isalpha() == False:
								first = first[:-1]+"."
							else:
								first = first+"."
							sent.append(clause+"@@@@"+first)
						else:
							sent.append("(S "+clause+"@@@@"+" ".join(subsent))

					self.splittedCC = sent


				else:
					res = str1+"@@@@"+str2
					self.splittedCC.append(res)


	def rmSBAR(self):
		# input should only be the parse tree
		for inp in self.splittedCC:
			if "<EOS>" in inp:
				self.takeSBAR.append(inp)
				continue

			inp, _ = inp.split("@@@@")
			tree = inp.split(" ")
			#print(tree)
			ifSBAR = 0
			SBARencountered = False 
			SBAR = []
			SBARtree = []
			allSBAR = []
			allSBARtree = []
			rest = []
			resttree = []
			for token in tree:
				if "(SBAR" in token and SBARencountered == False:
					ifSBAR += 1
					SBARencountered = True
					continue
				if ifSBAR != 0:
					SBARtree.append(token)
					if "(" in token:
						number = list(token).count("(")
						ifSBAR = ifSBAR + number
						#print(number)
					elif ")" in token:
						SBAR.append(token)
						#print(SBAR)
						number = list(token).count(")")
						ifSBAR = ifSBAR - number
						if ifSBAR <= 0: #<0 or <= 0?
							allSBAR.append(SBAR)
							allSBARtree.append(SBARtree)
							ifSBAR = 0
							SBARencountered = False
							SBAR = []
							SBARtree = []

				elif ifSBAR == 0:
					resttree.append(token)
					if ")" in token:
						rest.append(token)

			output = []
			SBAR = []
			SBARtree = []
			for token in rest:
				number = 0 - list(token).count(")")
				token = token[:number]
				output.append(token)
				#print(token)
			for oneSBAR in allSBAR:
				sbar = []
				for token in oneSBAR:
					number = 0 - list(token).count(")")
					token = token[:number]
					sbar.append(token)
				SBAR.append(" ".join(sbar))
			for oneSBARtree in allSBARtree:
				SBARtree.append(" ".join(oneSBARtree))

			senttree = " ".join(resttree)
			sent = " ".join(output)

			if SBAR != []:
				res = senttree+"@@@@"+sent
				self.takeSBAR.append(res)

				for i, sbar in enumerate(SBAR): #write down all the SBARs
					first = (sbar.split(" ")[0]).lower()
					#print(first)
					if "he" in first or "she" in first or "they" in first or "who" in first or "what" in first: #manipulate the first word in SBAR
						res = SBARtree[i]+"@@@@"+sbar
						self.takeSBAR.append(res)
					else:
						temp = sbar.split(" ")[1:]
						rmFirst = " ".join(temp)
						temptree = SBARtree[i].split(" ")[2:]
						rmtree = " ".join(temptree)
						res = rmtree+"@@@@"+rmFirst
						self.takeSBAR.append(res)



			else:
				res = senttree+"@@@@"+sent
				self.takeSBAR.append(res)

	def rmPP(self):
		for inp in self.takeSBAR:
			if "<EOS" in inp:
				self.removedPP.append(inp)
				continue
			inp = inp.split("@@@@")[0]

			cleaned = []
			#print(inp)
			check = inp.split(" ")[0]
			if ")" in check:
				temp = inp.split(" ")[1:]
				inp = " ".join(temp)

			parseTreeStr = inp
			parseTree = parseTreeStr.split(" ")
			ifPP = 0
			PPencountered = False
			for token in parseTree:
				if "(PP" in token and PPencountered == False:
					ifPP += 1
					PPencountered = True
					continue
				if ifPP != 0:
					if "(" in token:
						number = list(token).count("(")
						ifPP = ifPP + number
						#print(number)
					elif ")" in token:
						number = list(token).count(")")
						ifPP = ifPP - number
						if ifPP < 0:
							ifPP = 0
					#print(ifPP)
				elif ifPP == 0:
					if ")" in token:
						cleaned.append(token)
				#print(ifPP)
			output = []
			for token in cleaned:
				number = 0 - list(token).count(")")
				token = token[:number]
				output.append(token)
				#print(token)
			res = " ".join(output)
			self.removedPP.append(res)

	def rmPunc(self):
		for line in self.removedPP:
			if "<EOS>" in line:
				self.removedPunc.append(line)
				continue

			line = line.replace(",","")
			line = line.replace(".","")
			line = line.replace("'","")
			line = line.replace(":","")
			line = line.replace(";","")
			line = line.strip()
			if line == "":
				continue
			elif len(line.split(" ")) > 0 and line.split(" ")[-1].isalpha() == True:
				line = line+"."
			elif len(line.split(" ")) > 0 and line.split(" ")[-1].isalpha() == False:
				temp = line.split(" ")[:-1]
				line = " ".join(temp)+"."

			if len(line.split(" ")) > 0 and line.split(" ")[0][0].islower():
				line = line.split(" ")[0].title()+" "+" ".join(line.split(" ")[1:])
				#print(line)

			self.removedPunc.append(line)	
			#print(line)
			#f.write(line+"\n")


#inp = "(ROOT (S (S (VP (VBG Fleeing) (PP (IN from) (NP (DT the) (VBN perceived) (NN threat))))) (, ,) (NP (PRP she)) (VP (VBZ runs) (PP (PP (IN off) (NP (DT the) (NN train))) (, ,) (PP (IN through) (NP (DT the) (JJ deserted) (NN subway) (NN station))) (, ,) (CC and) (PP (IN onto) (NP (NP (DT the) (NN street)) (SBAR (WHADVP (WRB where)) (S (NP (PRP she)) (VP (VBZ gets) (S (S (VP (VBN attacked) (PP (IN in) (NP (DT a) (JJ dark) (NN alley))) (PP (IN by) (NP (NP (DT the) (VBG quacking) (NN maniac)) (, ,) (SBAR (WHNP (WP who)) (S (ADVP (RB brutally)) (VP (VBZ stabs) (NP (PRP her)) (PP (IN in) (NP (DT the) (NN leg)))))))))) (CC and) (S (VP (VBG slashes) (NP (PRP$ her) (NNS hands) (CC and) (NNS arms)) (SBAR (IN as) (S (NP (PRP she)) (VP (VBZ tries) (S (VP (TO to) (VP (VB defend) (NP (PRP herself)))))))))))))))))) (. .)));Fleeing from the perceived threat, she runs off the train, through the deserted subway station, and onto the street where she gets attacked in a dark alley by the quacking maniac, who brutally stabs her in the leg and slashes her hands and arms as she tries to defend herself."
inp = "Fleeing from the perceived threat, she runs off the train, through the deserted subway station, and onto the street where she gets attacked in a dark alley by the quacking maniac, who brutally stabs her in the leg and slashes her hands and arms as she tries to defend herself."
f = open("parsetree.txt","w")
inp2 = "I am going to school."
parser = parseLine()
parser.getTree(inp)
parser.getTree(inp2)
print("###################After getting parsed tree...####################")
print(parser.parsedSent)
print("\n")
for line in parser.parsedSent:
	f.write(line+"\n")
parser.splitCC()
print("###################After spliting CC...####################")
print(parser.splittedCC)
print("\n")
parser.rmSBAR()
print("###################After taking SBAR...####################")
print(parser.takeSBAR)
print("\n")
print("###################After removing PP...####################")
parser.rmPP()
print(parser.removedPP)
print("\n")
parser.rmPunc()
print("###################After removing punctuation...####################")
print(parser.removedPunc)