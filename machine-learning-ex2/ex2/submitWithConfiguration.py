from numpy import *
from os import remove
from os.path import isfile
from urllib.request import urlopen
from urllib.parse import urlencode
from pickle import load, dump
import json

def submitWithConfiguration(conf):

	parts = parts_(conf)

	print('== Submitting solutions | %s...' % conf['itemName'])

	tokenFile = 'token.pkl'
	if isfile(tokenFile):
		email, token = load(open(tokenFile, 'rb'))
		email, token = promptToken(email, token, tokenFile)
	else:
		email, token = promptToken('', '', tokenFile)

	if token == '':
		print('!! Submission Cancelled')
		return

	try:
		response = submitParts(conf, email, token, parts)
	except Exception as e:
		print(
			'!! Submission failed: unexpected error: %s'
			% e)
		print('!! Please try again later.')
		return

	if 'message' in response:
		print('!! Submission failed: %s' % response['message'])
	else:
		showFeedback(parts, response)
		dump((email, token), open(tokenFile, 'wb'))

def promptToken(email, existingToken, tokenFile):
	if email != '' and existingToken != '':
		prompt = sprintf(
			'Use token from last successful submission (%s)? (Y/n): ',
			 email)
		reenter = input(prompt)

		if reenter == '' or reenter == 'Y' or reenter == 'y':
			token = existingToken
			return email, token
		else:
			remove(tokenFile)

	email = input('Login (email address): ')
	token = input('Token: ')
	return email, token

def isValidPartOptionIndex(partOptions, i):
	return i != '' and 1 <= i and i <= size(partOptions)

def submitParts(conf, email, token, parts):
	body = makePostBody(conf, email, token, parts)
	submissionUrl = submissionUrl_()
	params = urlencode({'jsonBody': body}).encode()
	responseBody = urlopen(submissionUrl, data=params)
	return json.loads(responseBody.read().decode())

def makePostBody(conf, email, token, parts):
	bodyStruct = {}
	bodyStruct['assignmentSlug'] = conf['assignmentSlug']
	bodyStruct['submitterEmail'] = email
	bodyStruct['secret'] = token
	bodyStruct['parts'] = makePartsStruct(conf, parts)

	opt = {}
	opt['Compact'] = 1
	return json.dumps(bodyStruct).encode()

def makePartsStruct(conf, parts):
	partsStruct = {}
	for part in parts:
		partId = part['id']
		fieldName = makeValidFieldName(partId)
		outputStruct = {}
		outputStruct['output'] = conf['output'](partId)
		partsStruct[fieldName] = outputStruct
	return partsStruct

def parts_(conf):
	parts = []
	for partArray in conf['partArrays']:
		part = {}
		part['id'] = partArray[0]
		part['sourceFiles'] = partArray[1]
		part['name'] = partArray[2]
		parts.append(part)
	return parts

def showFeedback(parts, response):
	print('== ')
	print('== %43s | %9s | %-s' % ('Part Name', 'Score', 'Feedback'))
	print('== %43s | %9s | %-s' % ('---------', '-----', '--------'))
	for part in parts:
		score = ''
		partFeedback = ''
		partFeedback = response['partFeedbacks'][makeValidFieldName(part['id'])]
		partEvaluation = response['partEvaluations'][makeValidFieldName(part['id'])]
		score = sprintf('%d / %3d', partEvaluation['score'], partEvaluation['maxScore'])
		print('== %43s | %9s | %-s' % (part['name'], score, partFeedback))
	evaluation = response['evaluation']
	totalScore = sprintf('%d / %d', evaluation['score'], evaluation['maxScore'])
	print('==                                   --------------------------------')
	print('== %43s | %9s | %-s' % ('', totalScore, ''))
	print('== ')

###############################################################################
#
# Service configuration
#
###############################################################################
def submissionUrl_():
	return 'https://www-origin.coursera.org/api/onDemandProgrammingImmediateFormSubmissions.v1'

# ============================== HELPERS ==============================

def sprintf(formatSpec, *args):
    "emulates (part of) Octave sprintf function"
    if len(args) == 1:
        args = args[0]
        if type(args) == tuple:
            args = args[0]
    if type(args) == ndarray:
        return ''.join(formatSpec % e for e in args.ravel('F'))
    else:
        return formatSpec % args
        
def makeValidFieldName(str_):
	return str_