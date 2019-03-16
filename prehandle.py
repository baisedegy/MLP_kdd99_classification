
#coding:utf-8
 
import numpy as np
import csv
global label_list 
 
def preHandel_data():
    source_file='/Users/macbookpro/Downloads/corrected.csv'
    handled_file='KDD_testdata.csv'
    data_file=open(handled_file,'w',newline='')
    with open(source_file,'r') as data_source:
        csv_reader=csv.reader(data_source)
        csv_writer=csv.writer(data_file)
        count=0   
        for row in csv_reader:
            temp_line=np.array(row)
            
            temp_line[1]=handleProtocol(row)   
            temp_line[2]=handleService(row)    
            temp_line[3]=handleFlag(row)       
            temp_line[41]=handleLabel1(row) 
            temp_line=np.append(temp_line,handleLabel2(row))   
            csv_writer.writerow(temp_line)
            count+=1
           
            print(count,'status:',temp_line[1],temp_line[2],temp_line[3],temp_line[41],temp_line[42])
        data_file.close()
 

def find_index(x,y):
    for i in range(len(y)): 
        if y[i]==x:
            return i
 

def handleProtocol(input):
    protocol_list=['tcp','udp','icmp']
    if input[1] in protocol_list:
        return find_index(input[1],protocol_list)
    else:
        return 0

def handleService(input):
        service_list=['aol','auth','bgp','courier','csnet_ns','ctf','daytime','discard','domain','domain_u',
                 'echo','eco_i','ecr_i','efs','exec','finger','ftp','ftp_data','gopher','harvest','hostnames',
                 'http','http_2784','http_443','http_8001','imap4','IRC','iso_tsap','klogin','kshell','ldap',
                 'link','login','mtp','name','netbios_dgm','netbios_ns','netbios_ssn','netstat','nnsp','nntp',
                 'ntp_u','other','pm_dump','pop_2','pop_3','printer','private','red_i','remote_job','rje','shell',
                 'smtp','sql_net','ssh','sunrpc','supdup','systat','telnet','tftp_u','tim_i','time','urh_i','urp_i',
                 'uucp','uucp_path','vmnet','whois','X11','Z39_50']
        if input[2] in service_list:
                return find_index(input[2],service_list)
        else:
                return 0

def handleFlag(input):
    flag_list=['OTH','REJ','RSTO','RSTOS0','RSTR','S0','S1','S2','S3','SF','SH']
    if input[3] in flag_list:
        return find_index(input[3],flag_list)
    else:
            return 0

def handleLabel1(input):
    #label_list=['normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.', 'smurf.',
    # 'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.', 'ipsweep.', 'land.', 'ftp_write.',
    # 'back.', 'imap.', 'satan.', 'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.',
    # 'spy.', 'rootkit.']
    global label_list  
    if input[41] in label_list:
        return 0
    else:
        return 1
def handleLabel2(input):
    #label_list=['normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.', 'smurf.',
    # 'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.', 'ipsweep.', 'land.', 'ftp_write.',
    # 'back.', 'imap.', 'satan.', 'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.',
    # 'spy.', 'rootkit.']
    global label_list  
    if input[41] in label_list:
        return 1
    else:
        return 0
 
if __name__=='__main__':
    label_list=['normal.'] 
    preHandel_data()