import re

def tokenise_idiom( phrase ) :
    
    return 'ID' + re.sub( r'[\s|-]', '', phrase ).lower() + 'ID'

if __name__ == '__main__' :
    print( tokenise_idiom( 'big fish' ) )
    print( tokenise_idiom( 'alta-costura' ) )
    print( tokenise_idiom( 'pastoralemã' ) )
    assert tokenise_idiom( 'big fish' ) == 'IDbigfishID'
    assert tokenise_idiom( 'alta-costura' ) == 'IDaltacosturaID'
    assert tokenise_idiom( 'pão-duro' ) == 'IDpãoduroID'
    assert tokenise_idiom( 'pastoralemão' ) == 'IDpastoralemãoID'
    print( "All good" )
    
