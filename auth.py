import sqlalchemy as sql
from sqlalchemy import Table, Column, Integer, String, ForeignKey, MetaData
from sqlalchemy.exc import IntegrityError
import os


USERS_DIR = "USERS/"

meta = MetaData()
engine = sql.create_engine('sqlite:///mlTool.db', echo=True)

userTable = Table('Users', meta,
	Column('username', String, primary_key=True),
	Column('password', String),
	Column('emailID', String),
	Column('firstName', String),
	Column('lastName', String),
	Column('address', String),
	Column('city', String),
	Column('country', String),
	Column('postalCode', Integer),
	Column('aboutMe', String),
	)
meta.create_all(engine)

def addUser(userData):				# for sign up
	conn = engine.connect()
	username = userData['username'].lower()
	emailID = userData['emailID'].lower()
	password = userData['password']
	q = "INSERT into Users(username, password, emailID) values('" + username + "','" + password + "','" + emailID + "');"
	try:
		conn.execute(q)
	except IntegrityError:
		print("NOOOOO")
		return False
	print("YEEEES")
	os.makedirs(USERS_DIR+username)
	os.makedirs(USERS_DIR+username+"/datasets")
	os.makedirs(USERS_DIR+username+"/projects")
	# f = open(USERS_DIR+"metadata.txt", "w")
	return True


def checkUser(userData):		# for login
	conn = engine.connect()
	username = userData['username'].lower()
	password = userData['password']
	q = "SELECT * FROM Users WHERE username='" + username + "' AND password='" + password + "';"
	res = list(conn.execute(q))
	if res:
		return True
	else:
		return False


def checkUsername(userData):		# for sign up username
	print("8=D\n"*10, userData)
	username = userData["username"].lower()
	conn = engine.connect()
	q = "SELECT username FROM Users WHERE username='" + username + "';"
	res = list(conn.execute(q))
	# print("8=D\n"*5, res)
	if res:
		return "0"
	return "1"


def details(username):
	conn = engine.connect()
	username = username.lower()
	print(username)
	res = conn.execute("SELECT * FROM Users WHERE username='" + username + "';")
	res = list(res)
	print("HERE:", res)
	return list(res[0])


def updateDetails(username, newDetails):
	q = "UPDATE Users SET "
	for i in newDetails:
		if i == "postalCode":
			q += (i + "=" + str(newDetails[i]) +",")
		else:
			q += (i + "='" + str(newDetails[i]) +"',")
	q = q[:-1]
	q += " WHERE username='" + username.lower() + "';"
	print("Update query:", q)
	conn = engine.connect()
	conn.execute(q)
	print("CHECKING...")
	res = conn.execute("SELECT * FROM Users WHERE username='" + username + "';")
	print(list(res))
	return True
