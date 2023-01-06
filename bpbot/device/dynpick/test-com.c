// test-com.c
#define 	DEBUGSS	0

#include	<stdio.h>
#include	<fcntl.h>
#include	<time.h>
#include	<termios.h>
#include	<string.h>

#define true		1
#define false		0

int SetComAttr(int fdc);

int main()
	{
	int			status;
	FILE			*fd;
	int			fdc;
	char			fname[64];
	char			devname[64];
	char			str[256];
	unsigned short	data[6];
	int			comNo;
	int			tick;
	int			clk, clkb, clkb2, clk0;
	int			tw;
	int			num;
	int			n;


	fd = NULL;
	fdc = -1;

start :
	// COM�|�[�g���I�[�v��
	// printf("Enter COM port > ");
	// scanf("%d", &comNo);
	// printf("Open /dev/ttyUSB%d\n", comNo);

	// sprintf(devname, "/dev/ttyUSB%d", comNo);
	sprintf(devname, "/dev/ttyUSB0");
	fdc = open(devname, O_RDWR | O_NOCTTY | O_NONBLOCK);
	if (fdc < 0)
		goto over;

	// �T���v�����O�����𓾂�
	// tw = 16;
	tw = 100;
	// printf("Enter sampling time (ms) > ");
	// scanf("%d", &tw);
	printf("Sampling time = %d ms\n", tw);

	// printf("Enter File name > ");
	// char fname;
	// fname = "/home/hlab/bpbot/data/force/out_c.txt";
	// scanf("%s", fname);
	fd = fopen("/home/hlab/bpbot/data/force/out_c.txt", "w");
	if (!fd)
		goto over;

	// COM�|�[�g�̃{�[���[�g����ݒ�
	SetComAttr(fdc);

	// �f�[�^��ǂݏo��
	printf("=== record data ===\n");
	clk0 = clock() / (CLOCKS_PER_SEC / 1000);
	clkb = 0;
	clkb2 = 0;
	num = 0;

	// �P�f�[�^���N�G�X�g�i���񕪁j
	write(fdc, "R", 1);

	init_keyboard();

	while (true)
		{
		// �T���v�����O���������҂�
		while (true)
			{
			clk = clock() / (CLOCKS_PER_SEC / 1000) - clk0;

			if (clk >= clkb + tw)
				{
				clkb = clk / tw * tw;
				break;
				}
			}

		// �P�f�[�^���N�G�X�g�i���񕪁j
		write(fdc, "R", 1);

		// �P�f�[�^�𓾂�
		n = read(fdc, str, 27);	
		if (n < 27)
			{
//			printf ("=== error ! n = %d ===\n", n);
			goto skip;
			}

		sscanf(str, "%1d%4hx%4hx%4hx%4hx%4hx%4hx",
			 &tick, &data[0], &data[1], &data[2], &data[3], &data[4], &data[5]);

		sprintf(str, "%05d,%d,%05d,%05d,%05d,%05d,%05d,%05d\n",
			clk / tw * tw, tick,
			data[0], data[1], data[2], data[3], data[4], data[5]);

		fprintf(fd, str);
		num++;

skip :
		if (clk >= 10000)
			break;

		// �R���\�[���ɊԈ����\��
		if (clk >= clkb2 + 1000)
			{
			printf(str);
			if (kbhit() &&
				readch() == '.')
				break;
			clkb2 = clk / 1000 * 1000;
			}
		}

over1 :
	close_keyboard();
over :
	if (fd)
		{
		fclose(fd);
		fd = NULL;
		}

	if (fdc >= 0)
		{
		close(fdc);
		fdc = -1;
		}

	printf ("=== num = %d ===\n", num);

	printf("exit (y / n) ? > ");
	scanf("%s", str);
	if (str[0] == 'y')
		{
//		exit(0);
		}
	else
		{
		goto start;
		}
	}


int SetComAttr(int fdc)
	{
	int			n;

	struct termios	term;


	// �{�[���[�g����ݒ�
	n = tcgetattr(fdc, &term);
	if (n < 0)
		goto over;

	bzero(&term, sizeof(term));

	term.c_cflag = B921600 | CS8 | CLOCAL | CREAD;
	term.c_iflag = IGNPAR;
	term.c_oflag = 0;
	term.c_lflag = 0;/*ICANON;*/
 
	term.c_cc[VINTR]    = 0;     /* Ctrl-c */ 
	term.c_cc[VQUIT]    = 0;     /* Ctrl-? */
	term.c_cc[VERASE]   = 0;     /* del */
	term.c_cc[VKILL]    = 0;     /* @ */
	term.c_cc[VEOF]     = 4;     /* Ctrl-d */
	term.c_cc[VTIME]    = 0;
	term.c_cc[VMIN]     = 0;
	term.c_cc[VSWTC]    = 0;     /* '?0' */
	term.c_cc[VSTART]   = 0;     /* Ctrl-q */ 
	term.c_cc[VSTOP]    = 0;     /* Ctrl-s */
	term.c_cc[VSUSP]    = 0;     /* Ctrl-z */
	term.c_cc[VEOL]     = 0;     /* '?0' */
	term.c_cc[VREPRINT] = 0;     /* Ctrl-r */
	term.c_cc[VDISCARD] = 0;     /* Ctrl-u */
	term.c_cc[VWERASE]  = 0;     /* Ctrl-w */
	term.c_cc[VLNEXT]   = 0;     /* Ctrl-v */
	term.c_cc[VEOL2]    = 0;     /* '?0' */

//	tcflush(fdc, TCIFLUSH);
	n = tcsetattr(fdc, TCSANOW, &term);
over :
	
	return (n);
	}

