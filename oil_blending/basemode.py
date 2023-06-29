import math
class Distill():
	def __init__(self,temp):
		self.temp = temp
		self.percent = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1]
		self.valid = sum([1 for n in self.temp if n != float('inf')])
class Model():

	def __init__(self, oilA, oilB ):
		self.distilA = oilA
		self.distilB = oilB
		

	def blend(self, fractionA):
		"""
		This function performs the actual calculation
		type fractionA: float (between 0 and 1)
		rtype profile: List[float]
		"""
		profile = []
		self.fracA = fractionA
		for tempA, tempB in zip(self.distilA.temp, self.distilB.temp):

			# case 1: both A and B are still distilling
			if tempA != float('inf') and tempB != float('inf'):
				blend = round(self.fracA * tempA + (1-self.fracA) * tempB, 2)
				profile.append(blend)

			# case 2: both A and B stoped distilling
			elif tempA == float('inf') and tempB == float('inf'):
				blend.append(float('inf'))

			# case 3: one of A and B stoped distilling
			else:
				# current percent left of the blend, lower bondary
				percRemain = 1 - self.distilA.percent[len(profile)]
				# print(f'percRemain: {percRemain}, tempA: {tempA}, tempB: {tempB}')

				# When A stopped distilling, B continues
				if tempA == float('inf'):
					# % left of B with respect to the temperature before blending
					remain = 1 - self.distilB.percent[self.distilB.temp.index(tempB)]
					#  % left with respect to the temperature after blending
					bRemain = (1-self.fracA) * remain + self.fracA *(1-self.distilA.percent[self.distilA.valid-1])
					if bRemain <= percRemain:
						profile.append(tempB) #this temperature might higher than necssary

				# When B stopped distilling, A continues
				else:
					# % left of A with respect to the temperature before blending
					remain = 1 - self.distilA.percent[self.distilA.temp.index(tempA)]
					#  % left with respect to the temperature after blending
					aRemain = self.fracA * remain + (1-self.fracA) * (1-self.distilA.percent[self.distilB.valid-1])
					# print(f'aRemain: {aRemain}, bValid: {self.distilA.percent[self.distilB.valid-1]}')
					if aRemain <= percRemain: 
						profile.append(tempA) #this temperature might higher than necssary
		while len(profile)< len(self.distilA.percent):
			profile.append(float('inf'))

		return profile



if __name__ == '__main__':
	#A is BG past Avg(Feb 04, 2008), B is AWB past Avg(Jun 29, 2010)
	compA = Distill([28.0, 52.7, 78.7, 117.9, 160.2, 207.9, 264.8, 324.9, 393.3, 478.7, 571.4, 619.2, 685.2])
	compB = Distill([34.2, 46.8, 82.5, 235.4, 350.8, 426.4, 498.3, 582.4, 658.7, 708.9, float('inf'), float('inf'), float('inf')])

	m = Model(compA, compB)
	print(m.blend(0.8))
	assert m.blend(0) == compB.temp
	assert m.blend(1) == compA.temp
	