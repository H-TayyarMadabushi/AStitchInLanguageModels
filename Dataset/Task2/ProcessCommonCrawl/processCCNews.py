import os
import re
import sys
import csv
import time
import pickle
import html2text    
import subprocess
import multiprocessing

from tqdm          import tqdm
from collections   import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize   

sys.stderr.flush()
sys.stdout.flush()


## Outside class to speed up multithreading
def process_single_file( file_name, file_index, output_path, idioms, return_dict ) :

    this_scratch = os.path.join( output_path, 'scratch', str( file_index ) )
    try : 
        os.makedirs( this_scratch )
    except FileExistsError :
        pass

    file_name.split( ' ' )[-1]
    get_url = 'https://commoncrawl.s3.amazonaws.com/' + file_name.split( ' ' )[-1]

    file_path   = os.path.join( this_scratch, file_name.split( '/' )[-1] )
    wget_out    = os.path.join( this_scratch, 'wget.out'                 )
    get_command = 'wget -O ' + file_path + ' --output-file=' + wget_out + ' ' + get_url 

    if not os.path.isfile( file_path ) :  
        print( str( file_index ) +  "--> Downloading file ... ", end="", file=sys.stderr ); sys.stderr.flush()
        os.system( get_command )
        print( str( file_index ) +  "--> Done.", file=sys.stderr ); sys.stderr.flush()
    
        
    grep_file = os.path.join( this_scratch, 'output.txt' ) 
    grep_command = 'zgrep -i "' + '\|'.join( idioms ) + '" ' + file_path + ' > ' + grep_file

    if not os.path.isfile( grep_file ) : 
        print( str( file_index ) +  "--> Starting Grep ... ", end="", file=sys.stderr ); sys.stderr.flush()
        os.system( grep_command )
        print( str( file_index ) +  "--> Done.", file=sys.stderr ); sys.stderr.flush()

    h = html2text.HTML2Text()
    h.ignore_links = True

    idiom_words     = list()
    for idiom in idioms :
        idiom_words += re.split( ' |-', idiom )
    idiom_words     = set( idiom_words )
    
    sentences       = list()
    idiom_sentences = list()
    counts          = defaultdict( int ) 
    num_lines = sum(1 for line in open( grep_file,'r', encoding='utf-8', errors='ignore' ) )
    with open( grep_file, 'r', encoding='utf-8', errors='ignore' ) as output :
        for url_text in tqdm( output, total=num_lines, desc=str( file_index ) + '--> Processing' ) :
            try :
                url_text = h.handle( url_text )
            except : 
                continue
            url_text       = re.sub( r'\n([^\n])', r' \1', url_text )
            line           = url_text ## Legacy
            
            line = url_text
            if len( line ) > 10000 :
                continue
            if len( line.split() ) < 5 :
                continue

            if 'By the end users/application, this report covers the following segments'.lower() in line.lower() :
                continue
                
            if line.count( "*" ) > 1 :
                continue
            if line.count( '|' ) > 1 :
                continue
            if line.count( ':' ) > 1 :
                continue
            if line.count( '\\' ) > 3 :
                continue
            if line.count( '>' ) > 0 :
                continue
            if line.count( '#' ) > 2 :
                continue
            if line.count( '/' ) > 3 :
                continue
            if line.count( '=' ) > 0 :
                continue
            if '<div' in line :
                continue
            if '\x01' in line :
                continue
            if ');' in line :
                continue
            if any( [ ord( i ) in [ 1 ] for i in line ] ) :
                continue
            
            line = re.sub( r'\[(.*?)\]\(.*?\)', r'\1 ', line )
            line = re.sub( r'^[^a-zA-Z0-9_"“]*', '', line )
            line = re.sub( r'\*\*', '', line )
            line = re.sub( r'".*?":\s*"(.*)"(,)*', r'\1', line )
            line = re.sub( r'\n+^', '', line ) 
            
            line.replace( '_', '-' )
            line.replace( '  ', ' ' )

            had_idiom = False
            for idiom in idioms :
                if idiom.lower() in line.lower() :
                    idiom_sentences.append( line )
                    had_idiom = True
                    break
            if had_idiom :
                continue

                
            if any( [ ( idiom_word.lower() in line.lower() ) for idiom_word in idiom_words ] ) :
                sentences.append( line )

        
        idiom_file      = os.path.join( output_path, str( file_index ) + 'idioms.txt' )
        idiom_sentences = list( set( idiom_sentences ) )

        for sentence in idiom_sentences :
            for idiom in idioms :
                if idiom.lower() in sentence.lower() :
                    counts[ idiom ] += 1
        
        with open( idiom_file, 'w' ) as write_file :
            try :
                write_file.write( '\n--DocBreak--\n'.join( idiom_sentences ) )
            except UnicodeEncodeError:
                print( "Did not write idiom_sentences, unicode error - will retain scratch!" )
                return_dict[ file_index ] = defaultdict( int )
                return
            
        print( str( file_index ) + " --> Wrote ", idiom_file )

        idiom_word_file = os.path.join( output_path, str( file_index ) + 'idiom_words.txt' )
        sentences       = list( set( sentences ) ) 
        with open( idiom_word_file, 'w' ) as write_file :
            try : 
                write_file.write( '\n--DocBreak--\n'.join( sentences ) )
            except UnicodeEncodeError:
                print( "Did not write sentences, unicode error!" )
                pass

        
        return_dict[ file_index ] = counts

        # print( "DEBUG", flush=True )
        os.system( 'rm -rf ' + this_scratch )
        
        return

        



class processCCNews :

    def __init__( self, info_path='data', output_path='output-no-git', processes=9 ) :
        
        self.info_path   = info_path
        self.output_path = output_path
        self.processes   = processes
        
        self._get_idiom_info()
        self._init_run_state()
        return

    def _get_status_file( self ) : return os.path.join( self.info_path, 'processCCNNews-status.pk3' )
    
    def _init_run_state( self ) :

        file_location = os.path.join( self.info_path, '2020-news-files.txt' )
        crawl_files   = open( file_location ).read().split( '\n' )
        crawl_files   = [ i for i in crawl_files if i != '' ]

        status_file = self._get_status_file()
        if os.path.isfile( status_file ) :
            self.run_status = pickle.load( open( status_file, 'rb' ) )
        else :
            self.run_status = {
                'crawl_files' : crawl_files     ,
                'done'        : list()          ,  # list of done
                'processing'  : list()          ,  # list of files being processed (multithreadable)
                'counts'      : dict()          ,
            }

            for idiom in self.idioms :
                self.run_status[ 'counts' ][ idiom ] = 0

        self.run_status[ 'processing' ] = list()

        self._save_status()
        
        return

    def _save_status( self ) :
        pickle.dump( self.run_status, open( self._get_status_file(), 'wb' ) )
        return
        

    def _read_idioms( self, language ) :
        idioms_path = None
        if language in [ 'en', 'pt' ] :
            idioms_path = os.path.join( self.info_path, language + '_idioms.csv' )
        else :
            raise Exception( "Unknown Language" )

        idioms = list()
        with open( idioms_path ) as csvfile :
               reader = csv.reader(csvfile)
               for row in reader:
                   idioms.append( row[0] )
        return idioms
        
    
    def _get_idiom_info( self ) :
        en_idioms = self._read_idioms( 'en' ) 
        pt_idioms = self._read_idioms( 'pt' )

        self.idioms = en_idioms + pt_idioms

        self.idiom_words = list()
        for idiom in self.idioms :
            self.idiom_words += re.split( ' |-', idiom )

        return



    def _get_file_to_process( self ) :
        ## This function cannot be called from a sub-process
        index_to_process = None
        file_to_process  = None
        for index in range( len( self.run_status[ 'crawl_files' ] ) ) :
            if index not in self.run_status[ 'processing' ] and index not in self.run_status[ 'done' ] :
                self.run_status[ 'processing' ].append( index )
                return False, index, self.run_status[ 'crawl_files' ][ index ]
        return True, None, None

    def _update_run_status( self, done_file_index, done_file_counts ) :
        for elem in done_file_counts :
            self.run_status[ 'counts' ][ elem ] += done_file_counts[ elem ]
        self.run_status[ 'processing' ].remove( done_file_index )
        self.run_status[ 'done'       ].append( done_file_index )
        self._save_status()
        return
        

    def _keep_running( self ) :
        todo = open( os.path.join( self.info_path, 'run.info' ) ).read().lstrip().rstrip().split( '\n' )[0]
        return todo.lower() == 'run'

    def _hard_stop( self ) : 
        todo = open( os.path.join( self.info_path, 'run.info' ) ).read().lstrip().rstrip().split( '\n' )[0]
        return todo.lower() == 'hardstop'
    
    def _multithreaded_process( self ) :

        if self.processes == 1 :
            done = False
            while( not done and self._keep_running() ) :
                return_dict = dict()
                done, file_index, file_name  = self._get_file_to_process()
                if not done : 
                    process_single_file( file_name, file_index, self.output_path, self.idioms, return_dict )
                    this_counts = return_dict[ file_index ]
                    self._update_run_status( file_index, this_counts )
            return
            
        
        self.jobs = list()
        manager     = multiprocessing.Manager()
        return_dict = manager.dict()
        done        = False
        while( not done and self._keep_running() ) :
            for i in range( len( self.jobs ), self.processes ) :
                done, file_index, file_name  = self._get_file_to_process()
                if done :
                    break
                p = multiprocessing.Process(
                    target=process_single_file,
                    args=( file_name, file_index, self.output_path, self.idioms, return_dict )
                )
                self.jobs.append( (file_index, p ))
                p.start()

            running_jobs = list()
            for this_file_id, job in self.jobs :
                job.join(timeout=0)
                if job.is_alive() :
                    running_jobs.append( ( this_file_id, job ) )
                else :
                    try : 
                        this_counts = return_dict[ this_file_id ]
                        self._update_run_status( this_file_id, this_counts )
                    except KeyError :
                        print( "Error with thread ", this_file_id, flush=True )
                        pass
                    return_dict.pop( this_file_id, None )
            if len( running_jobs ) == len( self.jobs ) :
                time.sleep ( 5 )
                continue
            else :
                self.jobs = running_jobs


        if self._hard_stop() :
            for job in self.jobs :
                job[1].terminate()
            sys.exit()
        else :
            print( "Got request to stop, will wait for jobs to end ... ", flush=True )
            for this_file_id, job in self.jobs :
                job.join()
                this_counts = return_dict[ this_file_id ]
                self._update_run_status( this_file_id, this_counts )
                return_dict.pop( this_file_id, None )


    def process( self ) :
        try : 
            self._multithreaded_process()
        except KeyboardInterrupt:
            for job in self.jobs :
                job[1].terminate()

if __name__ == '__main__' :


    processor = processCCNews()
    processor.process()


    sys.exit()
        
