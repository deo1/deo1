__author__ = 'jfb_000'

import Tests
from HelperFunctions import *

# A Book object is a collection of contacts and methods for interacting with
# the contacts
class Book:

    # restricted variables
    __maxcontacts = 0  # int
    __contacts = {}  # dict of contacts only accessible through provided methods
    __numcontacts = 0  # int

    # public variables
    ownerid = ''  # owner of address book object

    # restricted methods
    def __init__(self, ownerid):
        self.ownerid = ownerid
        self.__maxcontacts = 200  # can not have more than 200 contacts in a single book
        self.__contacts = {}
        self.__numcontacts = self.__contacts.__len__()

    def __setMaxContacts(self, maxcontacts):
        self.__maxcontacts = maxcontacts
        print("New maximum amount of contacts is " + str(self.__maxcontacts))

    def __findContactKeysByValue(self, var, varname):
        keys = []
        for key, value in self.__contacts.items():
            if getattr(value, varname) == var:
                keys.append(key)
            else:
                pass
        return keys

    # public methods
    def addContact(self, firstname=None, lastname=None, phonenumber=None,
        emailaddress=None, street=None, city=None, state=None, country=None):
        # TODO condition for not a single argument set
        # if:
        #     pass

        contactobj = self.Contact(firstname, lastname, phonenumber, emailaddress, street, city, state, country)
        contactkey = (str(contactobj.firstname) + str(contactobj.lastname) + str(contactobj.phonenumber) +
                        str(contactobj.emailaddress) + str(contactobj.street) + str(contactobj.city) +
                        str(contactobj.state) + str(contactobj.country))

        # TODO check for key collision (does this even matter? it will just rewrite the value at worst)

        self.__contacts[contactkey] = contactobj

        # print("Contact " + contactobj.firstname + " " + contactobj.lastname + " added!")

        # TODO grow max size under some condition
        # self.__setMaxContacts(300)

    def removeContact(self, contact):
        # TODO implement this method
        if contact:
            try:
                del self.__contacts[contact.firstname + " " + contact.lastName]
                print("Contact " + contact.firstname + " " + contact.lastName + " removed!")
            except KeyError:
                print("No contact removed, because contact not found!")

    def dispContacts(self):
        # TODO implement this method
        for key, value in self.__contacts.items():
            value.dispContact()
        pass

    def numberOfContacts(self):
        return len(self.__contacts)

    def maxContacts(self):
        return self.__maxcontacts

    # TODO this is fast -- figure out how to know key
    def findContactByKey(self, contactkey):
        try:
            return self.__contacts[contactkey]
        except KeyError:
            print("Contact not found!")

    # TODO this is slow because it's searching on the values in a dictionary
    # I don't know a better data structure to use right now
    def findContacts(self, firstname=None, lastname=None, phonenumber=None,
        emailaddress=None, street=None, city=None, state=None, country=None):
        keysdict = {}
        if firstname:
            keysdict['firstname'] = self.__findContactKeysByValue(firstname, 'firstname')
        if lastname:
            keysdict['lastname'] = self.__findContactKeysByValue(lastname, 'lastname')
        if phonenumber:
            keysdict['phonenumber'] = self.__findContactKeysByValue(phonenumber, 'phonenumber')
        if emailaddress:
            keysdict['emailaddress'] = self.__findContactKeysByValue(emailaddress, 'emailaddress')
        if street:
            keysdict['street'] = self.__findContactKeysByValue(street, 'street')
        if city:
            keysdict['city'] = self.__findContactKeysByValue(city, 'city')
        if state:
            keysdict['state'] = self.__findContactKeysByValue(state, 'state')
        if country:
            keysdict['country'] = self.__findContactKeysByValue(country, 'country')

        # find key(s) that match all the values
        commonkeylist = [] # used for storing all of the keys that match all search criteria
        commonkeylisttemp = [] # used to store initial common keys before for looping without altered by removal
        firstitem = True

        # look through all the returned matching keys for each property of a
        # contact
        for key, keylist in keysdict.items():

            # if the property had any matches
            if keylist:

                # build a list of contact keys that match all the search
                # property values
                if firstitem:
                    commonkeylisttemp = keylist.copy()
                    commonkeylist = keylist.copy()
                    firstitem = False
                else:
                    for contactkey in commonkeylisttemp:
                        if contactkey in keylist:
                            pass
                        else:
                            commonkeylist.remove(contactkey)
        return commonkeylist

    # inner class
    # TODO external methods to Book can modify contacts, this will break the key
    # what to do, if anything?
    class Contact:
        # TODO allow N phone numbers?
        firstname = ''
        lastname = ''
        phonenumber = ''
        emailaddress = ''
        street = ''
        city = ''
        country = ''

        def __init__(self, firstname=None, lastname=None, phonenumber=None,
                     emailaddress=None, street=None, city=None, state=None,
                     country=None):
            self.firstname = firstname
            self.lastname = lastname
            self.phonenumber = phonenumber # TODO validate on ###_###_####, sanitize to # only
            self.emailaddress = emailaddress # TODO validate based on _@_._
            self.street = street
            self.city = city
            self.state = state # TODO validate state to two letter enums, sanitize to all caps
            self.country = country # TODO only support US at first

        def dispContact(self):
            print("\nCONTACT INFO")
            print(" First Name: " + str(self.firstname) + "\n" +
                  " Last Name: " + str(self.lastname) + "\n" +
                  " Phone Number: " + str(self.phonenumber) + "\n" +
                  " Email Address: " + str(self.emailaddress) + "\n" +
                  " Street: " + str(self.street) + "\n" +
                  " City: " + str(self.city) + "\n" +
                  " State: " + str(self.state) + "\n" +
                  " Country: " + str(self.country) + "\n")

        def delLastName(self):
            # TODO this is broken because dict key
            self.lastname = ''

# testing
if __name__ == '__main__':

    # Book class
    bookobj = Book('jfb_000')

    # add first contact
    Tests.testbook(bookobj=bookobj, firstname='Jesse', lastname='Bowman',
        phonenumber='423.555.3443', emailaddress='jb@u.coolschool.edu',
        street='1506 12th Ave Unit 420', city='Seattle', state='WA',
        country='USA')

    # add second contact
    Tests.testbook(bookobj=bookobj, firstname='Clandor', lastname='Dinkles', phonenumber='555.341.3443',
                   emailaddress='tard@gmail.com', street='1414', city='Bristol', state='TN', country='USA')

    # add third contact with shared values
    Tests.testbook(bookobj=bookobj, firstname='Jesse\'s', lastname='GF', phonenumber='205.555.1123',
                   emailaddress='HotGirlfriend@gmail.com', street='', city='Seattle', state='WA',
                   country='USA')

    # add a fourth contact with different shared values (want to test searching by some matches and some not)
    Tests.testbook(bookobj=bookobj, firstname='Jesse', lastname='Other', phonenumber='423.555.3443',
                   emailaddress='jb@u.coolschool.edu', street='1506 12th Ave Unit 421', city='Seattle', state='WA',
                   country='USA')
    # add a sparse contact
    Tests.testbook(bookobj=bookobj, city='Seattle')

    # add the first value again
    Tests.testbook(bookobj=bookobj, firstname='Jesse', lastname='Bowman', phonenumber='423.555.3443',
                   emailaddress='jb@u.coolschool.edu', street='1506 12th Ave Unit 420', city='Seattle', state='WA',
                   country='USA')

    # display all contacts
    print("=== Printing All Contacts in " + bookobj.ownerid + "'s Address Book === ")
    bookobj.dispContacts()

    # # Contact class
    # Tests.testcontact('Jesse')
    # Tests.testcontact('Jesse', 'Bowman')
    # Tests.testcontact('Jesse', 'Bowman', '423.341.3443')
    # Tests.testcontact('Jesse', 'Bowman', '423.341.3443', 'jb@u.coolschool.edu')
    # Tests.testcontact('Jesse', 'Bowman', '423.341.3443', 'jb@u.coolschool.edu', '1506 12th Ave Unit 420')
    # Tests.testcontact('Jesse', 'Bowman', '423.341.3443', 'jb@u.coolschool.edu', '1506 12th Ave Unit 420', 'Seattle')
    # Tests.testcontact('Jesse', 'Bowman', '423.341.3443', 'jb@u.coolschool.edu', '1506 12th Ave Unit 420', 'Seattle', 'USA')
