# with reference to: https://medium.com/@lhartikk/a-blockchain-in-200-lines-of-code-963cc1cc0e54
# with reference to: https://www.youtube.com/watch?v=Lx9zgZCMqXE

from hashlib import sha256
from datetime import datetime

def main():
    # test chain
    N = 10
    chain = Blockchain(data={'from': None, 'to': 'bill g.', 'amount': sum(range(N)), 'currency': 'BTC'})
    for ii in range(N):
        data = {'from': 'bill g.', 'to': 'jesse', 'amount': ii, 'currency': 'BTC'}
        chain.mine_block(data)
    print(chain)

    print()
    print("Valid chain:")
    if chain.validate_chain(): print("Test Passed")
    else: print("Test Failed")

    chain2 = chain + chain
    print()
    print("Invalid chain:")
    if chain2.validate_chain() == False: print("Test Passed")
    else: print("Test Failed")

    print()
    print(chain.accounts())
    print("Accounts Balanced:")
    print(chain.accounts_balanced())

class Block:
    def __init__(self, index, hash_previous, data):
        self.__timestamp = datetime.utcnow().timestamp()
        self.index = index
        self.hash_previous = hash_previous
        self.data = data
        self.hash = self.__calculate_hash()

    def __repr__(self):
        members = [(k, str(v)) for k, v in vars(self).items() if not k.startswith('_')]
        printable = ['    {}: {}'.format(m[0], m[1]) for m in members]
        return '{}{}{}'.format('Block(\n', '\n'.join(printable), ')')

    def __calculate_hash(self):
        hashable = ''.join([str(val) for val in vars(self).items()]).encode('utf-8')
        return sha256(hashable).hexdigest()

class Blockchain:
    # TODO : distributed mining via HTTP servers

    def __init__(self, data=None, chain=None):
        if chain is None:
            self.__chain = [Block(index=0, hash_previous=None, data=data)]
        else:
            self.__chain = chain

    def __repr__(self):
        return '\n <-- '.join([str(block) for block in self.__chain])

    def __len__(self):
        return len(self.__chain)

    def __add__(self, other):
        return Blockchain(chain=(self.__chain + other.__chain))

    def last_block(self):
        return self.__chain[-1]

    def mine_block(self, data):
        # TODO : proof of work for mining
        # TODO : branching chain
        # TODO : race condition / branch winner resolution

        index_next = self.last_block().index + 1
        hash_previous = self.last_block().hash
        block_next = Block(index_next, hash_previous, data)
        self.__chain.append(block_next)

    def validate_last_block(self, chain=None):
        if chain is None: chain = self.__chain
        hash_match = chain[-2].hash == chain[-1].hash_previous
        index_match = chain[-2].index == chain[-1].index - 1
        return (hash_match and index_match)

    def validate_chain(self, chain=None, chatty=0):
        # TODO : maximum blockchain size is 5000 due to recursion limits

        if chain is None: chain = self.__chain
        if len(chain) == 1:
            return True
        else:
            return self.validate_last_block(chain) and self.validate_chain(chain[:-1])

    def accounts(self):
        accounts = {}
        for block in self.__chain:
            data = block.data
            if data['to'] in accounts.keys():
                accounts[data['to']] += data['amount']
            else:
                accounts[data['to']] = data['amount']

            if data['from'] in accounts.keys():
                accounts[data['from']] -= data['amount']
            else:
                accounts[data['from']] = -data['amount']

        accounts.pop(None)
        return accounts

    def accounts_balanced(self):
        for val in self.accounts().values():
            if val < 0: return False

        return True

if __name__ == '__main__':
    main()