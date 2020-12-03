#include <sstream>

#include "chessUtils.h"

using bitMap = uint64_t;

enum bitboardIndex
{
	occupiedSquares,			//0		00000000
	whiteKing,					//1		00000001
	whiteQueens,				//2		00000010
	whiteRooks,					//3		00000011
	whiteBishops,				//4		00000100
	whiteKnights,				//5		00000101
	whitePawns,					//6		00000110
	whitePieces,				//7		00000111

	separationBitmap,			//8
	blackKing,					//9		00001001
	blackQueens,				//10	00001010
	blackRooks,					//11	00001011
	blackBishops,				//12	00001100
	blackKnights,				//13	00001101
	blackPawns,					//14	00001110
	blackPieces,				//15	00001111

	lastBitboard,

	King = whiteKing,
	Queens,
	Rooks,
	Bishops,
	Knights,
	Pawns,
	Pieces,
	whites = occupiedSquares,
	blacks = separationBitmap,

	empty = occupiedSquares

};

enum tSquare: int								/*!< square name and directions*/
{
	A1,	B1,	C1,	D1,	E1,	F1,	G1,	H1,
	A2,	B2,	C2,	D2,	E2,	F2,	G2,	H2,
	A3,	B3,	C3,	D3,	E3,	F3,	G3,	H3,
	A4,	B4,	C4,	D4,	E4,	F4,	G4,	H4,
	A5,	B5,	C5,	D5,	E5,	F5,	G5,	H5,
	A6,	B6,	C6,	D6,	E6,	F6,	G6,	H6,
	A7,	B7,	C7,	D7,	E7,	F7,	G7,	H7,
	A8,	B8,	C8,	D8,	E8,	F8,	G8,	H8,
	squareNone,
	squareNumber=64,
	north=8,
	sud=-8,
	est=1,
	ovest=-1,
	square0=0
};
inline tSquare operator+(const tSquare d1, const tSquare d2) { return static_cast<tSquare>(static_cast<int>(d1) + static_cast<int>(d2)); }
inline tSquare operator-(const tSquare d1, const tSquare d2) { return static_cast<tSquare>(static_cast<int>(d1) - static_cast<int>(d2)); }
inline tSquare& operator++(tSquare& d) { d = static_cast<tSquare>(static_cast<int>(d) + 1); return d; }
inline tSquare& operator+=(tSquare& d1, const tSquare d2) { d1 = d1 + d2; return d1; }
inline tSquare& operator-=(tSquare& d1, const tSquare d2) { d1 = d1 - d2; return d1; }

enum eNextMove	// color turn. ( it's also used as offset to access bitmaps by index)
{
	whiteTurn = 0,
	blackTurn = blackKing - whiteKing
};

static inline tSquare firstOne(const bitMap b)
{
	return (tSquare)__builtin_ctzll(b);
}

static inline tSquare iterateBit(bitMap& b)
{
	const tSquare t = firstOne(b);
	b &= ( b - 1 );
	return t;

}

unsigned int turnOffset(bool myTurn) {
    return myTurn ? 0 : 41600;
}

unsigned int whiteFeature(unsigned int  piece, tSquare pSquare, tSquare ksq, eNextMove turn) {
    unsigned int f = piece + (10 * pSquare) + (640 * ksq) + turnOffset(turn == eNextMove::whiteTurn);
    return f;
}

unsigned int blackFeature(unsigned int  piece, tSquare pSquare, tSquare ksq, eNextMove turn) {
    unsigned int f = piece + (10 * (pSquare ^ 56)) + (640 * (ksq ^ 56))+ turnOffset(turn == eNextMove::blackTurn);
    return f;
}

unsigned int whiteFeaturePSQ(unsigned int  piece, tSquare pSquare, eNextMove turn) {
    unsigned int f = piece + (10 * pSquare) + 40960 + turnOffset(turn == eNextMove::whiteTurn);
    return f;
}

unsigned int blackFeaturePSQ(unsigned int  piece, tSquare pSquare, eNextMove turn) {
    unsigned int f = piece + (10 * (pSquare ^ 56)) + 40960 + turnOffset(turn == eNextMove::blackTurn);
    return f;
}


void createBlackFeatures(bitMap* bitboards, std::vector<unsigned int>& fl, eNextMove turn){
    bitboardIndex blackPow[10] = {
        blackQueens,
        blackRooks,
        blackBishops,
        blackKnights,
        blackPawns,
        whiteQueens,
        whiteRooks,
        whiteBishops,
        whiteKnights,
        whitePawns
    };
    
    tSquare bkSq = firstOne(bitboards[bitboardIndex::blackKing]);
    for(unsigned int piece = 0; piece < 10; ++piece) {
        
        bitMap b = bitboards[blackPow[piece]];
        while(b)
        {
            tSquare pieceSq = iterateBit(b);
            fl.push_back(blackFeature(piece, pieceSq, bkSq, turn));
			fl.push_back(blackFeaturePSQ(piece, pieceSq, turn));
        }
    }
}

void createWhiteFeatures(bitMap* bitboards, std::vector<unsigned int>& fl, eNextMove turn){
    bitboardIndex whitePow[10] = {
        whiteQueens,
        whiteRooks,
        whiteBishops,
        whiteKnights,
        whitePawns,
        blackQueens,
        blackRooks,
        blackBishops,
        blackKnights,
        blackPawns
    };

    tSquare wkSq = firstOne(bitboards[bitboardIndex::whiteKing]);
    for(unsigned int piece = 0; piece < 10; ++piece) {

        bitMap b = bitboards[whitePow[piece]];
        while(b)
        {
            tSquare pieceSq = iterateBit(b);
            fl.push_back(whiteFeature(piece, pieceSq, wkSq, turn));
			fl.push_back(whiteFeaturePSQ(piece, pieceSq, turn));
        }
    }
}

std::vector<unsigned int> parseFen(const std::string& fenStr) {
    bitMap bitboard[lastBitboard] = {0};
    eNextMove turn; 
    char token;
	tSquare sq = A8;
	std::istringstream ss(fenStr);

	ss >> std::noskipws;

	while ((ss >> token) && !std::isspace(token))
	{
		if (isdigit(token))
			sq += tSquare(token - '0'); // Advance the given number of files
		else if (token == '/')
			sq -= tSquare(16);
		else
		{
			switch (token)
			{
			case 'P':
				bitboard[whitePawns] |= 1ull<<sq;
				break;
			case 'N':
				bitboard[whiteKnights] |= 1ull<<sq;
				break;
			case 'B':
				bitboard[whiteBishops] |= 1ull<<sq;
				break;
			case 'R':
				bitboard[whiteRooks] |= 1ull<<sq;
				break;
			case 'Q':
				bitboard[whiteQueens] |= 1ull<<sq;
				break;
			case 'K':
				bitboard[whiteKing] |= 1ull<<sq;
				break;
			case 'p':
				bitboard[blackPawns] |= 1ull<<sq;
				break;
			case 'n':
				bitboard[blackKnights] |= 1ull<<sq;
				break;
			case 'b':
				bitboard[blackBishops] |= 1ull<<sq;
				break;
			case 'r':
				bitboard[blackRooks] |= 1ull<<sq;
				break;
			case 'q':
				bitboard[blackQueens] |= 1ull<<sq;
				break;
			case 'k':
				bitboard[blackKing] |= 1ull<<sq;
				break;
			}
			++sq;
		}
	}

	ss >> token;
	turn = (token == 'w' ? whiteTurn : blackTurn);

    std::vector<unsigned int> featureList;
    createWhiteFeatures(bitboard, featureList, turn);
    createBlackFeatures(bitboard, featureList, turn);
    return featureList;
}