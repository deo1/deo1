__author__ = 'jfb_000'

from AddressBook import Book

def testbook(bookobj, firstname=None, lastname=None, phonenumber=None, emailaddress=None, street=None, city=None,
                 state=None, country=None):
    print("----------------TEST-------------------")

    # create book
    print("This is " + bookobj.ownerid + "'s address book")
    print("The maximum amount of contacts is " + str(bookobj.maxContacts()))
    print("Number of contacts in address book: " + str(bookobj.numberOfContacts()))

    # add contact with all values
    bookobj.addContact(firstname, lastname, phonenumber, emailaddress, street, city, state, country)
    print("Number of contacts in address book: " + str(bookobj.numberOfContacts()))

    # find contact via phone number
    if phonenumber:
        testphonenumber = phonenumber
        contactkeylist = bookobj.findContacts(phonenumber=testphonenumber)
        if contactkeylist:
            print("The contact(s) with phone number " + testphonenumber + " is:")
            for key in contactkeylist:
                bookobj.findContactByKey(key).dispContact()
        else:
            print("No contact with the phone number " + testphonenumber + " was found.")

    # find contact via street and city
    if street and city:
        teststreet = street
        testcity = city
        contactkeylist = bookobj.findContacts(street=teststreet, city=testcity)
        if contactkeylist:
            print("The contact(s) with address " + teststreet + " " + testcity + " is:")
            for key in contactkeylist:
                bookobj.findContactByKey(key).dispContact()
        else:
            print("No contact with the address " + teststreet + " " + testcity + " was found.")

    # testemail = 'jfb@u.northwestern.edu'
    # contact = bookobj.findContact(email=testemail)
    # if contact:
    #     print("The contact with email " + testemail + " is " + contact.firstname + " " + contact.lastname)
    # else:
    #     print("No contact with the email " + testemail + " was found.")
    # contact = bookobj.findContactByName(newcontact.firstname, newcontact.lastname)
    # contact2 = bookobj.findContactByName('Jesse')
    # contact.dispContact()
    # bookobj.removeContact(contact2)
    # contact.delLastName()
    # bookobj.removeContact(contact)
    # print("Number of contacts in address book: " + str(bookobj.numberOfContacts()))
    # num = bookobj.maxContacts()
    # print("The maximum amount of contacts is " + str(bookobj.maxContacts()))

# def testcontact(firstname=None, lastname=None, phonenumber=None, emailaddress=None, street=None, city=None,
#                  country=None):
#     print("----------------TEST-------------------")
#     contactobj = Contact(firstname, lastname, phonenumber, emailaddress, street, city, country)
#     print("Contact's first name is " + contactobj.firstName)
#     if contactobj.lastname is not None:
#         print("Contact's last name is " + contactobj.lastname)
#     else:
#         print('No last name')
#     if contactobj.phonenumber is not None:
#         print("Contact's phone number is " + contactobj.phonenumber)
#     else:
#         print('No phone number')
#     if contactobj.emailaddress is not None:
#         print("Contact's email address is " + contactobj.emailaddress)
#     else:
#         print('No email address')
#     if contactobj.street is not None:
#         print("Contact's street is " + contactobj.street)
#     else:
#         print('No street')
#     if contactobj.city is not None:
#         print("Contact's city is " + contactobj.city)
#     else:
#         print('No city')
#     if contactobj.country is not None:
#         print("Contact's country is " + contactobj.country)
#     else:
#         print('No country')